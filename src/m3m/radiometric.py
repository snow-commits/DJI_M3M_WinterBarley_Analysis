"""Radiometric calibration of DJI Mavic 3 Multispectral captures.

The model follows DJI's documented per-image processing chain:

1. scale the raw digital numbers to a unit interval,
2. divide out the per-pixel vignetting polynomial reported in the XMP block,
3. subtract the black level and clamp at zero,
4. normalise by sensor gain and exposure time,
5. apply the per-band sensor-gain adjustment,
6. normalise by the downwelling-light-sensor irradiance,
7. multiply by a single scene scale derived from the reflectance panel.

Step 7 is the only step that needs an external reference. The panel is anchored
on the NIR band, whose panel ROI is not saturated, and the resulting scale is
applied to every band. That makes the output a *provisional* reflectance: the
absolute level is only as good as the panel's nominal reflectance, and the
per-band uncertainty is not propagated here. See the README for the limits.

Nothing in this module reads global configuration; every input is an explicit
parameter so the functions can be unit tested without a dataset.
"""

from __future__ import annotations

import csv
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from m3m.constants import BANDS, NODATA
from m3m.io import (
    find_band,
    missing_xmp_tags,
    read_dji_xmp,
    read_raster,
    require_complete_groups,
    scan_capture_groups,
    write_single_band,
)

OUTPUT_SUFFIX = "_REFLECTANCE_APPROX.tif"

_MANIFEST_FIELDS = [
    "group_id",
    "band",
    "source",
    "output",
    "status",
    "error",
    "valid_fraction",
    "saturated_fraction",
    "reflectance_min",
    "reflectance_median",
    "reflectance_mean",
    "reflectance_max",
]


class RadiometricError(ValueError):
    """Raised when a capture cannot be calibrated."""


@dataclass(frozen=True)
class PanelAnchor:
    """Result of anchoring the reflectance scale on the NIR panel."""

    scale: float
    panel_reflectance_nir: float
    panel_signal: float
    valid_fraction: float


@dataclass
class CalibrationSummary:
    """Aggregate outcome of a batch calibration run."""

    output_dir: Path
    manifest_path: Path
    groups: int
    completed: int
    failed: int


def _as_float(metadata: Mapping[str, str], key: str, *, source: Path | str) -> float:
    try:
        return float(metadata[key])
    except KeyError as exc:
        raise RadiometricError(f"{source}: XMP property {key!r} is missing") from exc
    except (TypeError, ValueError) as exc:
        raise RadiometricError(
            f"{source}: XMP property {key!r} is not numeric: {metadata[key]!r}"
        ) from exc


def dji_relative_signal(data: np.ndarray, metadata: Mapping[str, str]) -> np.ndarray:
    """Apply the DJI vignetting/signal normalisation to one capture.

    Returns an irradiance-normalised signal, still without an absolute scale.
    The input must be an integer array; the model is defined on raw DN values.
    """
    if not np.issubdtype(data.dtype, np.integer):
        raise RadiometricError(
            f"The DJI signal model needs integer digital numbers, got {data.dtype}"
        )

    bits = np.iinfo(data.dtype).bits
    full_scale = float(2**bits)
    image = data.astype(np.float32) / full_scale

    black_level = _as_float(metadata, "BlackLevel", source="capture") / full_scale
    gain = _as_float(metadata, "SensorGain", source="capture")
    exposure_seconds = _as_float(metadata, "ExposureTime", source="capture") / 1_000_000.0
    band_gain_adjustment = _as_float(metadata, "SensorGainAdjustment", source="capture")
    irradiance = _as_float(metadata, "Irradiance", source="capture")

    if gain <= 0 or exposure_seconds <= 0 or irradiance <= 0:
        raise RadiometricError("Gain, exposure time and irradiance must all be positive")

    try:
        coefficients = [
            float(value.strip()) for value in metadata["VignettingData"].split(",")
        ]
    except KeyError as exc:
        raise RadiometricError("XMP property 'VignettingData' is missing") from exc
    if len(coefficients) != 6:
        raise RadiometricError(
            f"Expected six DJI vignetting coefficients, got {len(coefficients)}"
        )

    center_x = _as_float(metadata, "CalibratedOpticalCenterX", source="capture")
    center_y = _as_float(metadata, "CalibratedOpticalCenterY", source="capture")

    rows, columns = np.indices(data.shape, dtype=np.float32)
    radius = np.hypot(columns - center_x, rows - center_y)

    vignette_factor = np.ones(data.shape, dtype=np.float32)
    radius_power = radius.copy()
    for coefficient in coefficients:
        vignette_factor += coefficient * radius_power
        radius_power *= radius

    if not np.all(np.isfinite(vignette_factor)) or np.any(vignette_factor <= 0):
        raise RadiometricError("DJI vignetting model produced invalid correction factors")

    corrected = np.maximum(image * vignette_factor - black_level, 0.0)
    corrected /= gain * exposure_seconds
    corrected *= band_gain_adjustment
    corrected /= irradiance
    return corrected


def inspect_panel_image(
    path: Path | str,
    band: str,
    roi: Sequence[int],
    saturation_dn: int,
) -> dict[str, Any]:
    """Describe one panel capture: ROI statistics, saturation and XMP properties."""
    path = Path(path)
    data, _ = read_raster(path)
    height, width = data.shape
    x_min, y_min, x_max, y_max = (int(value) for value in roi)
    if not (0 <= x_min < x_max <= width and 0 <= y_min < y_max <= height):
        raise ValueError(
            f"Panel ROI {tuple(roi)} lies outside {path.name} ({width}x{height})"
        )

    panel = data[y_min:y_max, x_min:x_max]
    saturated = panel >= saturation_dn
    metadata = read_dji_xmp(path)

    return {
        "band": band,
        "path": path,
        "width": width,
        "height": height,
        "dtype": str(data.dtype),
        "roi": (x_min, y_min, x_max, y_max),
        "dn_min": float(panel.min()),
        "dn_median": float(np.median(panel)),
        "dn_mean": float(panel.mean()),
        "dn_max": float(panel.max()),
        "saturated_fraction": float(saturated.mean()),
        "saturated_pixels": int(saturated.sum()),
        "roi_pixels": int(panel.size),
        "metadata": metadata,
    }


def describe_panel_report(report: Mapping[str, Any]) -> str:
    """Render :func:`inspect_panel_image` output as a human-readable block."""
    lines = [
        f"{report['band']} band: {Path(report['path']).name}",
        f"  image size / dtype: {report['width']} x {report['height']} / {report['dtype']}",
        f"  panel ROI (x0,y0,x1,y1): {tuple(report['roi'])}",
        "  ROI DN min/median/mean/max: "
        f"{report['dn_min']:.1f} / {report['dn_median']:.1f} / "
        f"{report['dn_mean']:.1f} / {report['dn_max']:.1f}",
        "  fraction at or above saturation: "
        f"{100 * report['saturated_fraction']:.2f}% "
        f"({report['saturated_pixels']}/{report['roi_pixels']})",
    ]
    metadata = report["metadata"]
    if metadata:
        lines.append("  DJI XMP:")
        lines.extend(f"    {key}: {value}" for key, value in metadata.items())
    else:
        lines.append("  DJI XMP: not found on this capture")
    return "\n".join(lines)


def write_roi_overlay(path: Path | str, roi: Sequence[int], output_path: Path | str) -> Path:
    """Write a PNG preview of a panel capture with its ROI outlined.

    Uses the Agg backend so it is safe on headless machines and in CI.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    path = Path(path)
    image, _ = read_raster(path)
    x_min, y_min, x_max, y_max = (int(value) for value in roi)

    figure, axes = plt.subplots(figsize=(6, 6))
    axes.imshow(image, cmap="gray")
    axes.add_patch(
        Rectangle(
            (x_min, y_min),
            x_max - x_min,
            y_max - y_min,
            fill=False,
            edgecolor="red",
            linewidth=1.2,
        )
    )
    axes.set_xlim(max(0, x_min - 100), min(image.shape[1], x_max + 100))
    axes.set_ylim(min(image.shape[0], y_max + 100), max(0, y_min - 100))
    axes.set_title(path.name)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)
    return output_path


def calibrate_nir_anchor(
    *,
    panel_dir: Path | str,
    panel_roi: Mapping[str, Sequence[int]],
    panel_reflectance: Mapping[str, float],
    saturation_dn: int,
    panel_band: str = "NIR",
) -> PanelAnchor:
    """Estimate the shared reflectance scale from the NIR panel capture."""
    if panel_band not in panel_roi:
        raise RadiometricError(
            f"No panel ROI for the anchor band {panel_band!r}; "
            f"configured bands: {', '.join(sorted(panel_roi)) or 'none'}"
        )
    if panel_band not in panel_reflectance:
        raise RadiometricError(f"No nominal panel reflectance for band {panel_band!r}")

    path = find_band(panel_dir, panel_band)[0]
    data, _ = read_raster(path)
    metadata = read_dji_xmp(path)
    missing = missing_xmp_tags(metadata)
    if missing:
        raise RadiometricError(
            f"{path.name}: XMP is missing {', '.join(missing)}"
        )

    x_min, y_min, x_max, y_max = (int(value) for value in panel_roi[panel_band])
    panel = data[y_min:y_max, x_min:x_max]
    valid = np.isfinite(panel) & (panel < saturation_dn)
    if not np.any(valid):
        raise RadiometricError(
            f"Every pixel in the {panel_band} panel ROI is saturated or invalid; "
            f"re-derive the ROI with 'm3m panel --show-roi'."
        )

    corrected = dji_relative_signal(data, metadata)
    panel_signal = float(corrected[y_min:y_max, x_min:x_max][valid].mean())
    if panel_signal <= 0:
        raise RadiometricError("Corrected panel signal is not positive; check the ROI")

    nominal = float(panel_reflectance[panel_band])
    return PanelAnchor(
        scale=nominal / panel_signal,
        panel_reflectance_nir=nominal,
        panel_signal=panel_signal,
        valid_fraction=float(valid.mean()),
    )


def check_cross_band_agreement(
    anchor: PanelAnchor,
    *,
    panel_dir: Path | str,
    panel_roi: Mapping[str, Sequence[int]],
    panel_reflectance: Mapping[str, float],
    saturation_dn: int,
    bands: Sequence[str] = ("R", "RE"),
) -> list[dict[str, Any]]:
    """Compare predicted panel reflectance against the nominal values.

    This is a diagnostic, not a validation: with no independent ground
    reflectance measurement it only shows how consistent the NIR-derived scale
    is with the panel's own certificate.
    """
    rows: list[dict[str, Any]] = []
    for band in bands:
        if band not in panel_roi or band not in panel_reflectance:
            continue
        path = find_band(panel_dir, band)[0]
        data, _ = read_raster(path)
        metadata = read_dji_xmp(path)
        x_min, y_min, x_max, y_max = (int(value) for value in panel_roi[band])
        roi_dn = data[y_min:y_max, x_min:x_max]
        valid = roi_dn < saturation_dn
        row: dict[str, Any] = {"band": band, "saturated": not bool(np.any(valid))}
        if not np.any(valid):
            rows.append(row)
            continue
        reflectance = dji_relative_signal(data, metadata) * anchor.scale
        predicted = float(reflectance[y_min:y_max, x_min:x_max][valid].mean())
        nominal = float(panel_reflectance[band])
        row.update(
            nominal=nominal,
            predicted=predicted,
            relative_difference_percent=(predicted - nominal) / nominal * 100.0,
        )
        rows.append(row)
    return rows


def collect_flight_groups(img_dir: Path | str) -> dict[str, dict[str, Path]]:
    """Return complete four-band capture groups, or raise listing the gaps."""
    groups = scan_capture_groups(img_dir)
    spectral_only = {
        group_id: {band: path for band, path in paths.items() if band in BANDS}
        for group_id, paths in groups.items()
    }
    return require_complete_groups(spectral_only)


def calibrate_and_write(
    path: Path | str,
    band: str,
    scale: float,
    output_dir: Path | str,
    *,
    saturation_dn: int,
    panel_reflectance_nir: float,
) -> dict[str, Any]:
    """Calibrate one capture and write it as a single-band float32 GeoTIFF."""
    path = Path(path)
    data, profile = read_raster(path)
    metadata = read_dji_xmp(path)
    missing = missing_xmp_tags(metadata)
    if missing:
        raise RadiometricError(f"{path.name}: XMP is missing {', '.join(missing)}")

    reflectance = dji_relative_signal(data, metadata) * scale
    saturated = data >= saturation_dn
    valid = np.isfinite(reflectance) & ~saturated
    if not np.any(valid):
        raise RadiometricError(f"{path.name}: no usable pixels after saturation masking")
    reflectance[~valid] = NODATA

    output_path = Path(output_dir) / f"{path.stem}{OUTPUT_SUFFIX}"
    write_single_band(
        output_path,
        reflectance,
        profile,
        nodata=NODATA,
        tags={
            "PRODUCT": "Provisional M3M reflectance image",
            "CALIBRATION": "DJI metadata normalisation anchored to the NIR panel",
            "PANEL_NIR_REFLECTANCE": f"{panel_reflectance_nir:.6f}",
            "NIR_SCALE_FACTOR": f"{scale:.8f}",
            "SATURATION_DN": f"{saturation_dn}",
            "NOTE": (
                "Per-image reflectance estimate; saturation marked as NoData; "
                "not orthorectified and not a multispectral orthomosaic."
            ),
        },
    )

    values = reflectance[valid]
    return {
        "band": band,
        "source": str(path),
        "output": str(output_path),
        "valid_fraction": float(valid.mean()),
        "saturated_fraction": float(saturated.mean()),
        "reflectance_min": float(values.min()),
        "reflectance_median": float(np.median(values)),
        "reflectance_mean": float(values.mean()),
        "reflectance_max": float(values.max()),
    }


def _output_dir_for(out_dir: Path, limit: int | None) -> Path:
    if limit is None:
        return out_dir / "radiometric"
    return out_dir / f"radiometric_pilot_{limit}"


def process_dataset(
    scale: float,
    *,
    img_dir: Path | str,
    out_dir: Path | str,
    saturation_dn: int,
    panel_reflectance_nir: float,
    limit: int | None = None,
    quiet: bool = False,
) -> CalibrationSummary:
    """Calibrate every capture group, resuming from any previous manifest."""
    groups = collect_flight_groups(img_dir)
    if limit is not None:
        groups = dict(list(groups.items())[:limit])

    output_dir = _output_dir_for(Path(out_dir), limit)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "radiometric_manifest.csv"
    manifest_exists = manifest_path.exists()

    def log(message: str) -> None:
        if not quiet:
            print(message, flush=True)

    log(f"Capture groups to process: {len(groups)}")
    log(f"Output directory: {output_dir}")
    log("Processing image by image; the dataset is never loaded into memory as a whole.")
    log("Note: these are per-image reflectances, not an orthomosaic.")

    completed = 0
    failed = 0
    with manifest_path.open("a", newline="", encoding="utf-8-sig") as manifest_file:
        writer = csv.DictWriter(manifest_file, fieldnames=_MANIFEST_FIELDS)
        if not manifest_exists:
            writer.writeheader()
            manifest_file.flush()

        for group_index, (group_id, paths) in enumerate(groups.items(), start=1):
            group_failed = False
            for band in BANDS:
                path = paths[band]
                output_path = output_dir / f"{path.stem}{OUTPUT_SUFFIX}"
                if output_path.exists():
                    completed += 1
                    continue

                try:
                    result = calibrate_and_write(
                        path,
                        band,
                        scale,
                        output_dir,
                        saturation_dn=saturation_dn,
                        panel_reflectance_nir=panel_reflectance_nir,
                    )
                    writer.writerow({"group_id": group_id, **result, "status": "ok", "error": ""})
                    completed += 1
                except Exception as exc:  # noqa: BLE001 - recorded, not swallowed
                    writer.writerow(
                        {
                            "group_id": group_id,
                            "band": band,
                            "source": str(path),
                            "output": str(output_path),
                            "status": "failed",
                            "error": repr(exc),
                            "valid_fraction": "",
                            "saturated_fraction": "",
                            "reflectance_min": "",
                            "reflectance_median": "",
                            "reflectance_mean": "",
                            "reflectance_max": "",
                        }
                    )
                    failed += 1
                    group_failed = True
                    log(f"FAILED {group_id} {band}: {exc}")
                manifest_file.flush()

            if group_index == 1 or group_index % 10 == 0 or group_index == len(groups):
                log(
                    f"progress: {group_index}/{len(groups)} groups; "
                    f"{completed} images done, {failed} failed"
                    + (f"; last failure in group {group_id}" if group_failed else "")
                )

    log(f"Finished: {completed} images done, {failed} failed")
    log(f"Manifest: {manifest_path}")
    return CalibrationSummary(
        output_dir=output_dir,
        manifest_path=manifest_path,
        groups=len(groups),
        completed=completed,
        failed=failed,
    )


def preview_one_flight(
    scale: float,
    *,
    img_dir: Path | str,
    out_dir: Path | str,
    saturation_dn: int,
    panel_reflectance_nir: float,
    quiet: bool = False,
) -> list[Path]:
    """Calibrate the first capture of each band without orthorectification.

    Useful for a fast sanity check of the scale factor before committing to a
    full dataset run.
    """
    output_dir = Path(out_dir) / "radiometric_preview"
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    for band in BANDS:
        path = find_band(img_dir, band)[0]
        data, profile = read_raster(path)
        metadata = read_dji_xmp(path)
        reflectance = dji_relative_signal(data, metadata) * scale
        saturated = data >= saturation_dn
        reflectance[saturated] = NODATA

        output_path = write_single_band(
            output_dir / f"{path.stem}{OUTPUT_SUFFIX}",
            reflectance,
            profile,
            nodata=NODATA,
            tags={
                "PRODUCT": "Provisional M3M reflectance preview",
                "CALIBRATION": "DJI per-image metadata, anchored to the NIR panel",
                "PANEL_NIR_REFLECTANCE": f"{panel_reflectance_nir:.6f}",
                "NIR_SCALE_FACTOR": f"{scale:.8f}",
                "NOTE": "Approximate NIR panel anchor; not orthorectified",
            },
        )
        written.append(output_path)

        values = reflectance[~saturated]
        if not quiet:
            print(
                f"  {band}: min={values.min():.4f}, median={np.median(values):.4f}, "
                f"mean={values.mean():.4f}, max={values.max():.4f}, "
                f"saturated={100 * saturated.mean():.3f}%, saved={output_path}",
                flush=True,
            )
    return written


__all__ = [
    "CalibrationSummary",
    "OUTPUT_SUFFIX",
    "PanelAnchor",
    "RadiometricError",
    "calibrate_and_write",
    "calibrate_nir_anchor",
    "check_cross_band_agreement",
    "collect_flight_groups",
    "describe_panel_report",
    "dji_relative_signal",
    "inspect_panel_image",
    "preview_one_flight",
    "process_dataset",
    "write_roi_overlay",
]
