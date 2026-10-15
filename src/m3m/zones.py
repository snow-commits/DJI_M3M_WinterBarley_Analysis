"""Within-field relative zones derived from vegetation index rasters.

The five classes are equal-frequency quantiles *within the analysed field and
capture date*. They are relative ranks, not agronomic thresholds: class 5 means
"greener than 80% of this field today", not "high yield" or "needs top-dressing".
"""

from __future__ import annotations

import csv
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import rasterio  # noqa: E402
from matplotlib.colors import BoundaryNorm, ListedColormap  # noqa: E402
from rasterio.enums import Resampling  # noqa: E402

from m3m.constants import (  # noqa: E402
    HISTOGRAM_BINS,
    INDEX_ORDER,
    INDEX_RANGES,
    INDEX_SPECS,
    RELATIVE_CLASS_NAMES,
)

CLASS_RASTER_TEMPLATE = "{prefix}_{name}_relative_5class.tif"
DEFAULT_QUANTILE_EDGES: tuple[float, ...] = (0.2, 0.4, 0.6, 0.8)

__all__ = [
    "CLASS_RASTER_TEMPLATE",
    "calculate_thresholds",
    "classify_raster",
    "create_class_map",
    "create_distribution_map",
    "index_raster_path",
    "require_index_rasters",
    "run",
]


def index_raster_path(index_dir: Path | str, prefix: str, name: str) -> Path:
    """Path of a continuous index raster written by :mod:`m3m.indices`."""
    return Path(index_dir) / f"{prefix}_{name.lower()}.tif"


def require_index_rasters(
    index_dir: Path | str, prefix: str, index_names: Sequence[str] = INDEX_ORDER
) -> dict[str, Path]:
    """Return the index raster paths, failing early with an actionable message.

    ``zones``, ``compare`` and ``sensitivity`` all consume what ``indices``
    produced. Without this check the user gets a driver-level "No such file"
    error that does not mention the missing step.
    """
    index_dir = Path(index_dir)
    paths = {name: index_raster_path(index_dir, prefix, name) for name in index_names}
    missing = [path for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"Missing index raster(s) in {index_dir}: "
            f"{', '.join(path.name for path in missing)}.\n"
            f"  Run `m3m indices --raster <reflectance_orthomosaic.tif>` first, "
            f"or pass --index-dir / --prefix to match where they were written."
        )
    return paths


def class_raster_path(index_dir: Path | str, prefix: str, name: str) -> Path:
    """Path of a relative-class raster written by :func:`run`."""
    return Path(index_dir) / CLASS_RASTER_TEMPLATE.format(prefix=prefix, name=name.lower())


def percentile_from_histogram(
    histogram: np.ndarray, q: float, low: float, high: float, bins: int = HISTOGRAM_BINS
) -> float:
    """Estimate a quantile from a fixed-range histogram."""
    total = int(histogram.sum())
    if total == 0:
        raise ValueError("Cannot classify an index raster without valid pixels.")
    rank = q * (total - 1)
    bin_index = int(np.searchsorted(np.cumsum(histogram), rank + 1, side="left"))
    width = (high - low) / bins
    return float(low + (bin_index + 0.5) * width)


def calculate_thresholds(
    path: Path | str,
    index_name: str,
    *,
    quantiles: Sequence[float] = DEFAULT_QUANTILE_EDGES,
    bins: int = HISTOGRAM_BINS,
) -> list[float]:
    """Compute within-field quantile thresholds by streaming the raster once."""
    try:
        low, high = INDEX_RANGES[index_name.lower()]
    except KeyError as exc:
        raise ValueError(
            f"Unknown index {index_name!r}; expected one of {', '.join(INDEX_RANGES)}"
        ) from exc

    histogram = np.zeros(bins, dtype=np.int64)
    with rasterio.open(path) as src:
        for _, window in src.block_windows(1):
            values = src.read(1, window=window, masked=True).compressed()
            if values.size:
                counts, _ = np.histogram(values, bins=bins, range=(low, high))
                histogram += counts
    return [percentile_from_histogram(histogram, q, low, high, bins) for q in quantiles]


def classify_raster(
    path: Path | str,
    output_path: Path | str,
    index_name: str,
    thresholds: Sequence[float],
    *,
    bins: int = HISTOGRAM_BINS,
) -> dict[int, int]:
    """Write a uint8 relative-class raster and return the pixel count per class."""
    counts = {value: 0 for value in range(1, len(RELATIVE_CLASS_NAMES) + 1)}
    with rasterio.open(path) as src:
        profile = src.profile.copy()
        profile.update(
            count=1,
            dtype="uint8",
            nodata=0,
            compress="deflate",
            predictor=2,
            tiled=True,
            blockxsize=512,
            blockysize=512,
            BIGTIFF="IF_SAFER",
        )
        with rasterio.open(output_path, "w", **profile) as dst:
            dst.set_band_description(1, f"{index_name.upper()} relative quintile class")
            dst.update_tags(
                1,
                INDEX_NAME=index_name.upper(),
                CLASSIFICATION=(
                    "Five equal-frequency quantiles within this study field; "
                    "relative classes only, not agronomic thresholds"
                ),
                CLASS_CODES=(
                    "1=Lowest relative; 2=Low; 3=Middle; 4=High; "
                    "5=Highest relative; 0=NoData"
                ),
                QUANTILE_THRESHOLDS=";".join(f"{t:.8f}" for t in thresholds),
                THRESHOLD_METHOD=f"{bins}-bin histogram approximation",
            )
            for _, window in src.block_windows(1):
                source = src.read(1, window=window, masked=True)
                valid = ~np.ma.getmaskarray(source)
                result = np.zeros(source.shape, dtype=np.uint8)
                if np.any(valid):
                    values = source.data[valid]
                    classes = np.digitize(values, thresholds, right=False) + 1
                    result[valid] = classes.astype(np.uint8, copy=False)
                    for class_code in counts:
                        counts[class_code] += int(np.count_nonzero(classes == class_code))
                dst.write(result, 1, window=window)
            dst.build_overviews([2, 4, 8, 16], Resampling.nearest)
            dst.update_tags(ns="rio_overview", resampling="nearest")
    return counts


def _index_titles() -> dict[str, str]:
    return {
        name.lower(): f"{name}  |  {INDEX_SPECS[name]['formula']}" for name in INDEX_SPECS
    }


def create_distribution_map(
    index_dir: Path | str,
    boundary_path: Path | str,
    output_path: Path | str,
    *,
    prefix: str = "field",
    index_names: Sequence[str] = INDEX_ORDER,
    downsample: int = 8,
) -> Path:
    """Plot the continuous index rasters side by side with the field boundary."""
    index_dir = Path(index_dir)
    output_path = Path(output_path)
    titles = _index_titles()
    field = gpd.read_file(boundary_path)

    figure, axes = plt.subplots(
        1, len(index_names), figsize=(4 * len(index_names), 4.8), constrained_layout=True, squeeze=False
    )
    for ax, name in zip(axes.flat, index_names, strict=True):
        path = index_raster_path(index_dir, prefix, name)
        with rasterio.open(path) as src:
            out_shape = (
                max(1, src.height // downsample),
                max(1, src.width // downsample),
            )
            image = src.read(1, out_shape=out_shape, masked=True, resampling=Resampling.average)
            extent = (src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top)
            crs = src.crs
        values = image.compressed()
        if values.size == 0:
            raise ValueError(f"No valid map pixels found in {path}")
        vmin, vmax = np.percentile(values, (2, 98))
        plotted = ax.imshow(
            image, extent=extent, origin="upper", cmap="RdYlGn", vmin=vmin, vmax=vmax
        )
        field.to_crs(crs).boundary.plot(ax=ax, color="#202020", linewidth=0.8)
        ax.set_title(titles[name.lower()], fontsize=8.5, pad=5)
        ax.set_xlabel("Easting (m)", fontsize=7.5)
        ax.set_ylabel("Northing (m)", fontsize=7.5)
        ax.tick_params(labelsize=6.5)
        ax.set_aspect("equal")
        colorbar = figure.colorbar(plotted, ax=ax, fraction=0.042, pad=0.02)
        colorbar.set_label("Index value (P2-P98 stretch)", fontsize=7)
        colorbar.ax.tick_params(labelsize=6.5)

    figure.suptitle(
        "Vegetation Index Distribution\n"
        "Continuous index rasters; the display stretch affects visualization only",
        fontsize=11.5,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)
    return output_path


def create_class_map(
    index_dir: Path | str,
    boundary_path: Path | str,
    output_path: Path | str,
    *,
    prefix: str = "field",
    index_names: Sequence[str] = INDEX_ORDER,
    downsample: int = 8,
) -> Path:
    """Plot the relative-class rasters side by side with the field boundary."""
    index_dir = Path(index_dir)
    output_path = Path(output_path)
    colors = ["#a50026", "#f46d43", "#fee08b", "#a6d96a", "#1a9850"]
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(np.arange(0.5, 6.5, 1), cmap.N)
    titles = {name.lower(): f"{name} relative quintiles" for name in INDEX_SPECS}
    field = gpd.read_file(boundary_path)

    figure, axes = plt.subplots(
        1, len(index_names), figsize=(4 * len(index_names), 4.8), constrained_layout=True, squeeze=False
    )
    for ax, name in zip(axes.flat, index_names, strict=True):
        path = class_raster_path(index_dir, prefix, name)
        with rasterio.open(path) as src:
            out_shape = (
                max(1, src.height // downsample),
                max(1, src.width // downsample),
            )
            image = src.read(1, out_shape=out_shape, masked=True, resampling=Resampling.nearest)
            extent = (src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top)
            crs = src.crs
        ax.imshow(
            image, extent=extent, origin="upper", cmap=cmap, norm=norm, interpolation="nearest"
        )
        field.to_crs(crs).boundary.plot(ax=ax, color="#202020", linewidth=0.8)
        ax.set_title(titles[name.lower()], fontsize=8.5, pad=5)
        ax.set_xlabel("Easting (m)", fontsize=7.5)
        ax.set_ylabel("Northing (m)", fontsize=7.5)
        ax.tick_params(labelsize=6.5)
        ax.set_aspect("equal")
        mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        colorbar = figure.colorbar(
            mappable,
            ax=ax,
            fraction=0.05,
            pad=0.02,
            ticks=[1, 2, 3, 4, 5],
            boundaries=np.arange(0.5, 6.5, 1),
        )
        colorbar.ax.set_yticklabels(["Lowest", "Low", "Middle", "High", "Highest"])
        colorbar.set_label("Within-field relative rank", fontsize=7)
        colorbar.ax.tick_params(labelsize=6.5)

    figure.suptitle(
        "Relative Index Zones\n"
        "Five equal-frequency classes per index; relative ranking only, "
        "not agronomic thresholds",
        fontsize=11.5,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)
    return output_path


def run(
    index_dir: Path | str,
    boundary_path: Path | str,
    *,
    prefix: str = "field",
    index_names: Sequence[str] = INDEX_ORDER,
    bins: int = HISTOGRAM_BINS,
    make_figures: bool = True,
) -> list[dict[str, Any]]:
    """Classify every index into within-field quintiles and summarise the areas."""
    index_dir = Path(index_dir)

    reference_path = require_index_rasters(index_dir, prefix, index_names)[index_names[0]]
    with rasterio.open(reference_path) as reference:
        pixel_area_m2 = abs(
            reference.transform.a * reference.transform.e
            - reference.transform.b * reference.transform.d
        )

    rows: list[dict[str, Any]] = []
    for name in index_names:
        source_path = index_raster_path(index_dir, prefix, name)
        thresholds = calculate_thresholds(source_path, name, bins=bins)
        counts = classify_raster(
            source_path, class_raster_path(index_dir, prefix, name), name, thresholds, bins=bins
        )
        total = sum(counts.values())
        for class_code, class_name in enumerate(RELATIVE_CLASS_NAMES, start=1):
            pixels = counts[class_code]
            rows.append(
                {
                    "index": name.upper(),
                    "class_code": class_code,
                    "relative_class": class_name,
                    "lower_threshold": (
                        "-inf" if class_code == 1 else f"{thresholds[class_code - 2]:.8f}"
                    ),
                    "upper_threshold": (
                        "+inf"
                        if class_code == len(RELATIVE_CLASS_NAMES)
                        else f"{thresholds[class_code - 1]:.8f}"
                    ),
                    "pixel_count": pixels,
                    "area_ha": pixels * pixel_area_m2 / 10_000.0,
                    "area_percent": 100.0 * pixels / total if total else 0.0,
                    "threshold_method": f"{bins}-bin within-field histogram estimate",
                }
            )

    summary_path = index_dir / "vegetation_index_relative_classes.csv"
    with summary_path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    if make_figures:
        create_distribution_map(
            index_dir,
            boundary_path,
            index_dir / "vegetation_index_distribution.png",
            prefix=prefix,
            index_names=index_names,
        )
        create_class_map(
            index_dir,
            boundary_path,
            index_dir / "vegetation_index_relative_classes.png",
            prefix=prefix,
            index_names=index_names,
        )
    return rows
