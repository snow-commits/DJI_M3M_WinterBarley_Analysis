"""Generate a small synthetic M3M dataset.

This module exists so the toolkit can be exercised end to end - in CI, in a
fresh clone, or by a reviewer - without shipping any real flight data.

Two artefacts are produced:

``raw``
    Capture groups named ``<prefix>_<NNNN>_MS_<BAND>.TIF`` carrying the same DJI
    XMP properties as a real capture, plus reflectance-panel captures with a
    known panel ROI. Synthetic digital numbers are written as the *exact
    inverse* of the DJI signal model, so calibrating them back must return the
    chosen reflectance. That round trip is what the end-to-end test asserts.

``ortho``
    A five-band reflectance orthomosaic (Red, Green, NIR, RedEdge, Alpha) with
    band descriptions matching what :mod:`m3m.indices` expects, plus a field
    boundary GeoPackage.

The data is deterministic: a fixed seed yields byte-identical output, so tests
can assert on values rather than on shapes alone.
"""

from __future__ import annotations

import os
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from rasterio.errors import NotGeoreferencedWarning
from rasterio.transform import from_origin

from m3m.constants import BANDS, NODATA

DEFAULT_SHAPE = (240, 320)  # (height, width)
DEFAULT_PANEL_ROI = (40, 60, 120, 140)  # (x_min, y_min, x_max, y_max)
DEFAULT_FIELD_PREFIX = "synthetic"
DEFAULT_CRS = "EPSG:32633"

#: Full-scale factor of the model. The generator inverts the DJI chain, so the
#: chosen sensor parameters must keep synthetic digital numbers inside uint16.
FULL_SCALE = 65536.0
DEFAULT_SENSOR = {
    "BlackLevel": 1000.0,
    "SensorGain": 1.0,
    "ExposureTime": 1000.0,  # microseconds
    "SensorGainAdjustment": 1.0,
    "Irradiance": 1000.0,
}

#: Reflectance the generator aims for, per band and region. Kept below 0.8 so
#: the inverted digital numbers stay inside the uint16 range.
PANEL_REFLECTANCE = {"G": 0.493, "R": 0.494, "RE": 0.494, "NIR": 0.493}
SOIL_REFLECTANCE = 0.18
VEGETATION_REFLECTANCE = {"G": 0.09, "R": 0.05, "RE": 0.28, "NIR": 0.52}

_XMP_TEMPLATE = """<x:xmpmeta xmlns:x="adobe:ns:meta/">
 <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description xmlns:drone-dji="http://www.dji.com/drone-dji/1.0/"
{attributes}  />
 </rdf:RDF>
</x:xmpmeta>"""


@dataclass(frozen=True)
class RawDataset:
    """Paths and settings describing a generated raw dataset."""

    root: Path
    img_dir: Path
    panel_dir: Path
    groups: int
    shape: tuple[int, int]
    panel_roi: tuple[int, int, int, int]
    panel_reflectance: Mapping[str, float]
    saturation_dn: int
    scene_scale: float
    target_reflectance: Mapping[str, np.ndarray] | None = None


@dataclass(frozen=True)
class OrthomosaicDataset:
    """Paths describing a generated reflectance orthomosaic."""

    root: Path
    raster: Path
    boundary: Path
    shape: tuple[int, int]
    pattern: np.ndarray


def build_xmp(metadata: Mapping[str, Any]) -> str:
    """Render a DJI XMP block with the properties the toolkit reads."""
    attributes = "".join(f'   drone-dji:{key}="{value}"\n' for key, value in metadata.items())
    return _XMP_TEMPLATE.format(attributes=attributes)


def write_capture(
    path: Path | str,
    dn: np.ndarray,
    metadata: Mapping[str, Any],
    *,
    band_description: str | None = None,
) -> Path:
    """Write one uint16 capture with an embedded DJI XMP block."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    profile = {
        "driver": "GTiff",
        "count": 1,
        "dtype": "uint16",
        "width": dn.shape[1],
        "height": dn.shape[0],
        "compress": "deflate",
    }
    # Real M3M captures carry no geotransform, so neither do the synthetic ones.
    # Silence the resulting (expected) rasterio warning.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", NotGeoreferencedWarning)
        with rasterio.open(path, "w", **profile) as dst:
            dst.write(dn.astype(np.uint16), 1)
            if band_description:
                dst.set_band_description(1, band_description)
            dst.update_tags(ns="xml:XMP", **{"xml:XMP": build_xmp(metadata)})
    return path


def _vignette_factor(
    shape: tuple[int, int], center: tuple[float, float], coefficients: Sequence[float]
) -> np.ndarray:
    """Mirror of the DJI vignetting polynomial used to *create* synthetic data.

    The duplication is intentional: inverting the model here means the test
    asserts a real round trip through :mod:`m3m.radiometric`.
    """
    rows, columns = np.indices(shape, dtype=np.float64)
    radius = np.hypot(columns - center[0], rows - center[1])
    factor = np.ones(shape, dtype=np.float64)
    power = radius.copy()
    for coefficient in coefficients:
        factor += coefficient * power
        power *= radius
    return factor


def dn_from_reflectance(
    reflectance: np.ndarray,
    *,
    scale: float,
    vignette: np.ndarray,
    sensor: Mapping[str, float] = DEFAULT_SENSOR,
) -> np.ndarray:
    """Invert the DJI signal chain: target reflectance -> digital numbers."""
    signal = np.asarray(reflectance, dtype=np.float64) / scale
    signal = signal * float(sensor["Irradiance"])
    signal = signal / float(sensor["SensorGainAdjustment"])
    signal = signal * float(sensor["SensorGain"]) * float(sensor["ExposureTime"]) / 1_000_000.0
    signal = signal + float(sensor["BlackLevel"]) / FULL_SCALE
    signal = signal / vignette
    dn = signal * FULL_SCALE
    if np.any(dn >= 65535):
        raise ValueError(
            "Synthetic digital numbers overflow uint16. Lower the target reflectance, "
            "the irradiance or the exposure time."
        )
    return np.rint(dn)


def _field_pattern(shape: tuple[int, int], seed: int) -> tuple[np.ndarray, np.ndarray]:
    """A smooth vegetation gradient plus mild texture, both in [0, 1]."""
    rng = np.random.default_rng(seed)
    height, width = shape
    rows, columns = np.indices(shape, dtype=np.float64)
    gradient = 0.55 * (columns / max(1, width - 1)) + 0.45 * (rows / max(1, height - 1))
    texture = rng.normal(scale=0.04, size=shape)
    smooth_texture = np.zeros(shape, dtype=np.float64)
    smooth_texture[1:-1, 1:-1] = (
        texture[1:-1, 1:-1]
        + texture[:-2, 1:-1]
        + texture[2:, 1:-1]
        + texture[1:-1, :-2]
        + texture[1:-1, 2:]
    ) / 5.0
    pattern = np.clip(gradient + smooth_texture, 0.0, 1.0)
    return pattern, rng.random(shape)


def reflectance_for_band(band: str, pattern: np.ndarray) -> np.ndarray:
    """Blend soil and vegetation endmembers with the field pattern."""
    if band == "G":
        vegetation = VEGETATION_REFLECTANCE["G"]
    else:
        vegetation = VEGETATION_REFLECTANCE[band]
    return SOIL_REFLECTANCE + pattern * (vegetation - SOIL_REFLECTANCE)


def _write_rgb_frame(path: Path, pattern: np.ndarray) -> Path:
    """Write the true-colour companion frame (``*_D.JPG``) of a capture group.

    M3M capture folders interleave four spectral TIFFs with one RGB JPEG per
    group; the audit treats a group without its RGB frame as incomplete, so the
    synthetic dataset reproduces that layout. Uses Pillow, which matplotlib
    already depends on.
    """
    import matplotlib.image as mpimg

    red = (0.18 + 0.30 * pattern) * 255
    green = (0.22 + 0.36 * pattern) * 255
    blue = (0.16 + 0.22 * pattern) * 255
    rgb = np.stack([red, green, blue], axis=-1).clip(0, 255).astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    mpimg.imsave(path, rgb, format="jpg")
    return path


def generate_raw_dataset(
    root: Path | str,
    *,
    groups: int = 3,
    shape: tuple[int, int] = DEFAULT_SHAPE,
    panel_roi: tuple[int, int, int, int] = DEFAULT_PANEL_ROI,
    scene_scale: float = 1.0,
    vignette_coefficients: Sequence[float] = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    seed: int = 20260101,
    prefix: str = DEFAULT_FIELD_PREFIX,
    with_rgb: bool = True,
) -> RawDataset:
    """Write ``<root>/Dataset`` and ``<root>/Reflectance_Panel``."""
    root = Path(root)
    img_dir = root / "Dataset"
    panel_dir = root / "Reflectance_Panel"
    img_dir.mkdir(parents=True, exist_ok=True)
    panel_dir.mkdir(parents=True, exist_ok=True)

    height, width = shape
    center = (width / 2.0, height / 2.0)
    vignette = _vignette_factor(shape, center, vignette_coefficients)
    x_min, y_min, x_max, y_max = panel_roi
    if not (0 <= x_min < x_max <= width and 0 <= y_min < y_max <= height):
        raise ValueError(
            f"panel_roi {panel_roi} does not fit inside a {width}x{height} capture; "
            f"pass a panel_roi that lies inside the synthetic image."
        )

    pattern, _ = _field_pattern(shape, seed)
    targets = {band: reflectance_for_band(band, pattern) for band in BANDS}

    # Panels: uniform soil-like background with the calibrated panel in the ROI.
    for band in BANDS:
        panel_reflectance = np.full(shape, SOIL_REFLECTANCE, dtype=np.float64)
        panel_reflectance[y_min:y_max, x_min:x_max] = PANEL_REFLECTANCE[band]
        metadata = {
            "BandName": band,
            "BlackLevel": DEFAULT_SENSOR["BlackLevel"],
            "SensorGain": DEFAULT_SENSOR["SensorGain"],
            "ExposureTime": DEFAULT_SENSOR["ExposureTime"],
            "SensorGainAdjustment": DEFAULT_SENSOR["SensorGainAdjustment"],
            "Irradiance": DEFAULT_SENSOR["Irradiance"],
            "VignettingData": ",".join(str(value) for value in vignette_coefficients),
            "CalibratedOpticalCenterX": center[0],
            "CalibratedOpticalCenterY": center[1],
            "RtkFlag": 1,
            "GpsStatus": "rtk",
        }
        dn = dn_from_reflectance(
            panel_reflectance, scale=scene_scale, vignette=vignette
        )
        write_capture(
            panel_dir / f"{prefix}_0001_MS_{band}.TIF", dn, metadata, band_description=band
        )

    # Flight groups: one capture per band per group.
    for index in range(1, groups + 1):
        for band in BANDS:
            target = targets[band]
            metadata = {
                "BandName": band,
                "BlackLevel": DEFAULT_SENSOR["BlackLevel"],
                "SensorGain": DEFAULT_SENSOR["SensorGain"],
                "ExposureTime": DEFAULT_SENSOR["ExposureTime"],
                "SensorGainAdjustment": DEFAULT_SENSOR["SensorGainAdjustment"],
                "Irradiance": DEFAULT_SENSOR["Irradiance"],
                "VignettingData": ",".join(str(value) for value in vignette_coefficients),
                "CalibratedOpticalCenterX": center[0],
                "CalibratedOpticalCenterY": center[1],
                "RtkFlag": 1,
                "GpsStatus": "rtk",
            }
            dn = dn_from_reflectance(target, scale=scene_scale, vignette=vignette)
            write_capture(
                img_dir / f"{prefix}_{index:04d}_MS_{band}.TIF",
                dn,
                metadata,
                band_description=band,
            )
        if with_rgb:
            _write_rgb_frame(img_dir / f"{prefix}_{index:04d}_D.JPG", pattern)

    return RawDataset(
        root=root,
        img_dir=img_dir,
        panel_dir=panel_dir,
        groups=groups,
        shape=shape,
        panel_roi=panel_roi,
        panel_reflectance=dict(PANEL_REFLECTANCE),
        saturation_dn=65520,
        scene_scale=scene_scale,
        target_reflectance=targets,
    )


def generate_orthomosaic(
    root: Path | str,
    *,
    shape: tuple[int, int] = DEFAULT_SHAPE,
    pixel_size: float = 0.05,
    origin: tuple[float, float] = (400000.0, 5000000.0),
    seed: int = 20260101,
    prefix: str = DEFAULT_FIELD_PREFIX,
    crs: str = DEFAULT_CRS,
) -> OrthomosaicDataset:
    """Write a five-band reflectance orthomosaic plus its field boundary."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    raster_path = root / f"{prefix}_reflectance_orthomosaic.tif"
    boundary_path = root / f"{prefix}_boundary.gpkg"

    height, width = shape
    pattern, _ = _field_pattern(shape, seed)
    transform = from_origin(origin[0], origin[1], pixel_size, pixel_size)

    bands = {
        "Red": reflectance_for_band("R", pattern),
        "Green": reflectance_for_band("G", pattern),
        "NIR": reflectance_for_band("NIR", pattern),
        "RedEdge": reflectance_for_band("RE", pattern),
    }
    alpha = np.full(shape, 255, dtype=np.uint8)

    profile = {
        "driver": "GTiff",
        "count": 5,
        "dtype": "float32",
        "width": width,
        "height": height,
        "crs": crs,
        "transform": transform,
        "nodata": NODATA,
        "compress": "deflate",
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256,
    }
    with rasterio.open(raster_path, "w", **profile) as dst:
        for position, name in enumerate(("Red", "Green", "NIR", "RedEdge"), start=1):
            dst.write(bands[name].astype(np.float32), position)
            dst.set_band_description(position, name)
        dst.write(alpha, 5)
        dst.set_band_description(5, "Alpha")

    _write_boundary(boundary_path, transform, width, height, crs)

    return OrthomosaicDataset(
        root=root,
        raster=raster_path,
        boundary=boundary_path,
        shape=shape,
        pattern=pattern,
    )


def _write_boundary(
    path: Path,
    transform: Any,
    width: int,
    height: int,
    crs: str,
    *,
    inset_fraction: float = 0.05,
) -> Path:
    """Write a rectangular field boundary inset from the raster edges."""
    import geopandas as gpd
    from shapely.geometry import box

    inset_x = width * inset_fraction
    inset_y = height * inset_fraction
    x_min, y_max = transform * (inset_x, inset_y)
    x_max, y_min = transform * (width - inset_x, height - inset_y)
    frame = gpd.GeoDataFrame(
        {"name": ["synthetic field"]},
        geometry=[box(x_min, y_min, x_max, y_max)],
        crs=crs,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_file(path, driver="GPKG")
    return path


def _portable(config_dir: Path, target: Path) -> str:
    """Express *target* relative to the config file, so the two move together."""
    config_dir = Path(config_dir).resolve()
    target = Path(target).resolve()
    try:
        return Path(os.path.relpath(target, config_dir)).as_posix()
    except ValueError:
        # A relative path does not exist across Windows drives.
        return target.as_posix()


def write_config(
    path: Path | str,
    dataset: RawDataset,
    *,
    out_dir: str = "outputs",
    orthomosaic: OrthomosaicDataset | None = None,
) -> Path:
    """Write a ready-to-use ``m3m.toml`` pointing at the generated data.

    Paths are written relative to the config file's own directory, which is how
    :mod:`m3m.config` resolves them, so the config and the data stay together
    when the whole directory is moved.
    """
    path = Path(path)
    config_dir = path.parent
    lines = [
        "# Generated by `m3m synthetic`. Safe to delete.",
        "",
        "[paths]",
        f'img_dir = "{_portable(config_dir, dataset.img_dir)}"',
        f'panel_dir = "{_portable(config_dir, dataset.panel_dir)}"',
        f'out_dir = "{out_dir}"',
    ]
    if orthomosaic is not None:
        lines.append(f'boundary = "{_portable(config_dir, orthomosaic.boundary)}"')
    lines += [
        "",
        "[radiometric]",
        f"saturation_dn = {dataset.saturation_dn}",
        "",
        "[radiometric.panel_reflectance]",
    ]
    lines += [f"{band} = {value}" for band, value in dataset.panel_reflectance.items()]
    lines += ["", "[radiometric.panel_roi]"]
    roi = ", ".join(str(value) for value in dataset.panel_roi)
    lines += [f"{band} = [{roi}]" for band in BANDS]
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
