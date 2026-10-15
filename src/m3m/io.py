"""Raster, vector and DJI XMP input/output helpers.

Everything here is deliberately free of configuration lookups: functions take
explicit paths so they can be reused from tests, notebooks and the CLI alike.
"""

from __future__ import annotations

import re
import warnings
from collections import defaultdict
from collections.abc import Iterable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.errors import NotGeoreferencedWarning

from m3m.constants import BANDS, NODATA

# Raw M3M captures intentionally carry no geotransform, so every read of a
# capture raises this warning. It is expected here and would otherwise flood
# stderr on a full dataset run (1,200+ frames).
warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)

DJI_XMP_NAMESPACE = "http://www.dji.com/drone-dji/1.0/"
RDF_NAMESPACE = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"

#: XMP properties read out of every M3M capture.
XMP_TAGS: tuple[str, ...] = (
    "BandName",
    "BandFreq",
    "GpsStatus",
    "RtkFlag",
    "Irradiance",
    "SensorGain",
    "ExposureTime",
    "SensorGainAdjustment",
    "BlackLevel",
    "LS_status",
    "VignettingFlag",
    "VignettingData",
    "CalibratedOpticalCenterX",
    "CalibratedOpticalCenterY",
)

#: XMP properties without which the DJI signal model cannot be applied.
REQUIRED_XMP_TAGS: tuple[str, ...] = (
    "BlackLevel",
    "SensorGain",
    "ExposureTime",
    "SensorGainAdjustment",
    "Irradiance",
    "VignettingData",
    "CalibratedOpticalCenterX",
    "CalibratedOpticalCenterY",
)

#: Capture group identifier, e.g. ``DJI_202401150930_0007_MS_NIR.TIF`` -> ``0007``.
_GROUP_PATTERN = re.compile(r"_(\d{4})(?:_MS_|_D$)")
_BAND_PATTERN = re.compile(r"_MS_(G|R|RE|NIR)$", re.IGNORECASE)

_RASTER_SUFFIXES = {".tif", ".tiff"}
_RGB_SUFFIXES = {".jpg", ".jpeg"}

#: Band key used for the true-colour frame that accompanies each capture group.
RGB_KEY = "RGB"


__all__ = [
    "DJI_XMP_NAMESPACE",
    "RDF_NAMESPACE",
    "REQUIRED_XMP_TAGS",
    "RGB_KEY",
    "XMP_TAGS",
    "assert_same_grid",
    "extract_xmp_packet",
    "find_band",
    "iter_band_paths",
    "missing_xmp_tags",
    "read_boundary",
    "read_dji_xmp",
    "read_raster",
    "reflectance_profile",
    "require_complete_groups",
    "scan_capture_groups",
    "valid_mask",
    "write_single_band",
]


def read_boundary(path: Path | str) -> gpd.GeoDataFrame:
    """Read the field boundary as a GeoDataFrame."""
    boundary = gpd.read_file(path)
    if boundary.empty:
        raise ValueError(f"Boundary file contains no features: {path}")
    return boundary


def find_band(directory: Path | str, band: str) -> list[Path]:
    """Return every ``*_MS_<band>`` raster in *directory*, sorted by name.

    Matching is case-insensitive on both the band token and the file extension,
    so captures copied from a Windows card still resolve on Linux and macOS.
    """
    directory = Path(directory)
    if not directory.is_dir():
        raise NotADirectoryError(f"Not a directory: {directory}")
    suffix = f"_MS_{band.upper()}"
    files = sorted(
        path
        for path in directory.iterdir()
        if path.is_file()
        and path.suffix.lower() in _RASTER_SUFFIXES
        and path.stem.upper().endswith(suffix)
    )
    if not files:
        raise FileNotFoundError(
            f"No {band.upper()} band image (expected '*_MS_{band.upper()}.TIF') in {directory}"
        )
    return files


def read_raster(path: Path | str) -> tuple[np.ndarray, dict[str, Any]]:
    """Read band 1 of *path* and return ``(array, profile)``."""
    with rasterio.open(path) as src:
        data = src.read(1)
        profile = src.profile.copy()
    return data, profile


def extract_xmp_packet(raw: str | None) -> str | None:
    """Return the XML packet from a raw ``xml:XMP`` metadata value.

    GDAL serialises entries in its ``xml:*`` metadata domains as
    ``"<path>=<value>"``. Captures written by DJI itself read back as a bare
    packet starting with ``<``, but a file that has been through
    ``gdal_translate`` or written by a GDAL-based tool reads back as
    ``xml:XMP=<x:xmpmeta ...>``. Accepting both forms keeps the reader working
    on either provenance.
    """
    if not raw:
        return None
    text = raw.strip()
    if text.startswith("<"):
        return text
    start = text.find("<")
    return text[start:] if start != -1 else None


def read_dji_xmp(path: Path | str) -> dict[str, str]:
    """Return the DJI XMP properties embedded in a capture, as a plain dict."""
    with rasterio.open(path) as src:
        raw = src.tags(ns="xml:XMP").get("xml:XMP")

    xmp = extract_xmp_packet(raw)
    if not xmp:
        return {}

    try:
        root = ElementTree.fromstring(xmp)
    except ElementTree.ParseError as exc:
        raise ValueError(f"Malformed XMP block in {path}: {exc}") from exc

    description = root.find(f".//{{{RDF_NAMESPACE}}}Description")
    if description is None:
        return {}

    tags: dict[str, str] = {}
    for key, value in description.attrib.items():
        if key.startswith(f"{{{DJI_XMP_NAMESPACE}}}"):
            local_name = key.rsplit("}", 1)[-1]
            if local_name in XMP_TAGS:
                tags[local_name] = value
    return tags


def missing_xmp_tags(metadata: Mapping[str, str]) -> list[str]:
    """Return the required XMP properties absent from *metadata*."""
    return [name for name in REQUIRED_XMP_TAGS if name not in metadata]


def scan_capture_groups(directory: Path | str) -> dict[str, dict[str, Path]]:
    """Group raw captures by their four-digit capture id.

    Returns a mapping ``capture_id -> {band_key: path}`` where ``band_key`` is
    one of ``G``, ``R``, ``RE``, ``NIR`` or ``RGB`` (the JPEG companion frame).
    Groups are sorted by capture id.
    """
    directory = Path(directory)
    if not directory.is_dir():
        raise NotADirectoryError(f"Not a directory: {directory}")

    groups: dict[str, dict[str, Path]] = defaultdict(dict)
    for path in directory.iterdir():
        if not path.is_file():
            continue
        match = _GROUP_PATTERN.search(path.stem)
        if not match:
            continue
        group_id = match.group(1)
        suffix = path.suffix.lower()
        stem_upper = path.stem.upper()

        band_match = _BAND_PATTERN.search(stem_upper)
        if suffix in _RASTER_SUFFIXES and band_match:
            band = band_match.group(1).upper()
            key = band
        elif suffix in _RGB_SUFFIXES and stem_upper.endswith("_D"):
            key = RGB_KEY
        else:
            continue

        if key in groups[group_id]:
            raise ValueError(
                f"Duplicate {key} frame for capture group {group_id}: "
                f"{groups[group_id][key].name} and {path.name}"
            )
        groups[group_id][key] = path

    return {group_id: dict(bands) for group_id, bands in sorted(groups.items())}


def require_complete_groups(
    groups: Mapping[str, Mapping[str, Path]],
    *,
    bands: Sequence[str] = BANDS,
    max_examples: int = 10,
) -> dict[str, dict[str, Path]]:
    """Fail loudly when any capture group is missing a spectral band.

    Partial groups are the usual cause of silently incomplete orthomosaics, so
    this is an error rather than a warning.
    """
    expected = set(bands)
    incomplete = {
        group_id: sorted(expected - set(paths))
        for group_id, paths in groups.items()
        if set(paths) != expected
    }
    if incomplete:
        examples = "; ".join(
            f"{group_id}: missing {','.join(missing)}"
            for group_id, missing in list(sorted(incomplete.items()))[:max_examples]
        )
        raise ValueError(
            f"{len(incomplete)} capture group(s) are missing spectral bands; "
            f"stopping to avoid partial outputs. {examples}"
        )
    if not groups:
        raise ValueError("No capture groups found.")
    return {group_id: dict(paths) for group_id, paths in groups.items()}


def reflectance_profile(profile: Mapping[str, Any], *, nodata: float = NODATA) -> dict[str, Any]:
    """Derive a single-band float32 GeoTIFF profile from an input profile."""
    updated = dict(profile)
    updated.update(
        driver="GTiff",
        count=1,
        dtype="float32",
        nodata=nodata,
        compress="deflate",
        predictor=3,
    )
    return updated


def write_single_band(
    path: Path | str,
    array: np.ndarray,
    profile: Mapping[str, Any],
    *,
    nodata: float = NODATA,
    tags: Mapping[str, Any] | None = None,
    band_description: str | None = None,
) -> Path:
    """Write one float32 band with the standard compression and NoData settings."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", **reflectance_profile(profile, nodata=nodata)) as dst:
        dst.write(np.asarray(array, dtype=np.float32), 1)
        if band_description:
            dst.set_band_description(1, band_description)
        if tags:
            dst.update_tags(**{key: str(value) for key, value in tags.items()})
    return path


def valid_mask(dataset: rasterio.io.DatasetReader, values: np.ndarray) -> np.ndarray:
    """Finite values that are not the dataset's NoData sentinel."""
    mask = np.isfinite(values)
    if dataset.nodata is not None:
        mask &= values != dataset.nodata
    return mask


def assert_same_grid(
    reference: rasterio.io.DatasetReader, other: rasterio.io.DatasetReader
) -> None:
    """Raise when two rasters do not share width, height, transform and CRS."""
    if (reference.width, reference.height, reference.transform, reference.crs) != (
        other.width,
        other.height,
        other.transform,
        other.crs,
    ):
        raise ValueError(
            f"Raster grid mismatch: {other.name} does not align with {reference.name}"
        )


def iter_band_paths(paths: Mapping[str, Path], bands: Iterable[str] = BANDS) -> Iterator[tuple[str, Path]]:
    """Yield ``(band, path)`` in the canonical band order."""
    for band in bands:
        yield band, paths[band]
