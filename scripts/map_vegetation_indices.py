"""绘制选定的三个指数栅格，并生成田内相对五分位等级。"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
import numpy as np
import rasterio
from rasterio.enums import Resampling


INDEX_RANGES = {
    "ndvi": (-1.0, 1.0),
    "ndre": (-1.0, 1.0),
    "evi2": (-2.5 / 3.4, 1.25),
}
HISTOGRAM_BINS = 100_000
CLASS_NAMES = ["Lowest relative", "Low", "Middle", "High", "Highest relative"]


def percentile_from_histogram(histogram: np.ndarray, q: float, low: float, high: float) -> float:
    total = int(histogram.sum())
    if total == 0:
        raise ValueError("Cannot classify an index raster without valid pixels.")
    rank = q * (total - 1)
    bin_index = int(np.searchsorted(np.cumsum(histogram), rank + 1, side="left"))
    width = (high - low) / HISTOGRAM_BINS
    return float(low + (bin_index + 0.5) * width)


def calculate_thresholds(path: Path, index_name: str) -> list[float]:
    low, high = INDEX_RANGES[index_name]
    histogram = np.zeros(HISTOGRAM_BINS, dtype=np.int64)
    with rasterio.open(path) as src:
        for _, window in src.block_windows(1):
            values = src.read(1, window=window, masked=True).compressed()
            if values.size:
                counts, _ = np.histogram(values, bins=HISTOGRAM_BINS, range=(low, high))
                histogram += counts
    return [percentile_from_histogram(histogram, q, low, high) for q in (0.2, 0.4, 0.6, 0.8)]


def classify_raster(path: Path, output_path: Path, index_name: str, thresholds: list[float]) -> dict[int, int]:
    counts = {value: 0 for value in range(1, 6)}
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
                CLASSIFICATION="Five equal-frequency quantiles within this study field; relative classes only, not agronomic thresholds",
                CLASS_CODES="1=Lowest relative; 2=Low; 3=Middle; 4=High; 5=Highest relative; 0=NoData",
                QUANTILE_THRESHOLDS=";".join(f"{t:.8f}" for t in thresholds),
                THRESHOLD_METHOD=f"{HISTOGRAM_BINS}-bin histogram approximation",
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


def create_distribution_map(index_dir: Path, boundary_path: Path, output_path: Path) -> None:
    names = ["ndvi", "ndre", "evi2"]
    titles = {
        "ndvi": "NDVI  |  (NIR - Red) / (NIR + Red)",
        "ndre": "NDRE  |  (NIR - RedEdge) / (NIR + RedEdge)",
        "evi2": "EVI2  |  2.5 (NIR - Red) / (NIR + 2.4 Red + 1)",
    }
    field = gpd.read_file(boundary_path)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.8), constrained_layout=True)
    for ax, name in zip(axes, names):
        path = index_dir / f"winterbarley_{name}.tif"
        with rasterio.open(path) as src:
            factor = 8
            out_shape = (max(1, src.height // factor), max(1, src.width // factor))
            image = src.read(1, out_shape=out_shape, masked=True, resampling=Resampling.average)
            extent = (
                src.bounds.left,
                src.bounds.right,
                src.bounds.bottom,
                src.bounds.top,
            )
            crs = src.crs
        values = image.compressed()
        if values.size == 0:
            raise ValueError(f"No valid map pixels found in {path}")
        vmin, vmax = np.percentile(values, (2, 98))
        im = ax.imshow(image, extent=extent, origin="upper", cmap="RdYlGn", vmin=vmin, vmax=vmax)
        field.to_crs(crs).boundary.plot(ax=ax, color="#202020", linewidth=0.8)
        ax.set_title(titles[name], fontsize=8.5, pad=5)
        ax.set_xlabel("Easting (m)", fontsize=7.5)
        ax.set_ylabel("Northing (m)", fontsize=7.5)
        ax.tick_params(labelsize=6.5)
        ax.set_aspect("equal")
        cbar = fig.colorbar(im, ax=ax, fraction=0.042, pad=0.02)
        cbar.set_label("Index value (P2–P98 stretch)", fontsize=7)
        cbar.ax.tick_params(labelsize=6.5)

    fig.suptitle(
        "Winter Barley Vegetation Index Distribution\n"
        "Continuous index rasters; display stretch affects visualization only",
        fontsize=11.5,
    )
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def create_class_map(index_dir: Path, boundary_path: Path, output_path: Path) -> None:
    names = ["ndvi", "ndre", "evi2"]
    colors = ["#a50026", "#f46d43", "#fee08b", "#a6d96a", "#1a9850"]
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(np.arange(0.5, 6.5, 1), cmap.N)
    titles = {
        "ndvi": "NDVI relative quintiles",
        "ndre": "NDRE relative quintiles",
        "evi2": "EVI2 relative quintiles",
    }
    field = gpd.read_file(boundary_path)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.8), constrained_layout=True)
    for ax, name in zip(axes, names):
        path = index_dir / f"winterbarley_{name}_relative_5class.tif"
        with rasterio.open(path) as src:
            factor = 8
            out_shape = (max(1, src.height // factor), max(1, src.width // factor))
            image = src.read(1, out_shape=out_shape, masked=True, resampling=Resampling.nearest)
            extent = (src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top)
            crs = src.crs
        ax.imshow(image, extent=extent, origin="upper", cmap=cmap, norm=norm, interpolation="nearest")
        field.to_crs(crs).boundary.plot(ax=ax, color="#202020", linewidth=0.8)
        ax.set_title(titles[name], fontsize=8.5, pad=5)
        ax.set_xlabel("Easting (m)", fontsize=7.5)
        ax.set_ylabel("Northing (m)", fontsize=7.5)
        ax.tick_params(labelsize=6.5)
        ax.set_aspect("equal")
        mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        cbar = fig.colorbar(
            mappable,
            ax=ax,
            fraction=0.05,
            pad=0.02,
            ticks=[1, 2, 3, 4, 5],
            boundaries=np.arange(0.5, 6.5, 1),
        )
        cbar.ax.set_yticklabels(["Lowest", "Low", "Middle", "High", "Highest"])
        cbar.set_label("Within-field relative rank", fontsize=7)
        cbar.ax.tick_params(labelsize=6.5)

    fig.suptitle(
        "Relative Index Zones\n"
        "Five equal-frequency classes per index; relative ranking only, not agronomic thresholds",
        fontsize=11.5,
    )
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index-dir", type=Path, required=True)
    parser.add_argument("--boundary", type=Path, required=True)
    args = parser.parse_args()
    names = ["ndvi", "ndre", "evi2"]
    with rasterio.open(args.index_dir / "winterbarley_ndvi.tif") as reference:
        pixel_area_m2 = abs(
            reference.transform.a * reference.transform.e
            - reference.transform.b * reference.transform.d
        )

    rows: list[dict[str, object]] = []
    for name in names:
        source_path = args.index_dir / f"winterbarley_{name}.tif"
        thresholds = calculate_thresholds(source_path, name)
        class_path = args.index_dir / f"winterbarley_{name}_relative_5class.tif"
        counts = classify_raster(source_path, class_path, name, thresholds)
        total = sum(counts.values())
        for class_code, class_name in enumerate(CLASS_NAMES, start=1):
            pixels = counts[class_code]
            rows.append(
                {
                    "index": name.upper(),
                    "class_code": class_code,
                    "relative_class": class_name,
                    "lower_threshold": "-inf" if class_code == 1 else f"{thresholds[class_code - 2]:.8f}",
                    "upper_threshold": "+inf" if class_code == 5 else f"{thresholds[class_code - 1]:.8f}",
                    "pixel_count": pixels,
                    "area_ha": pixels * pixel_area_m2 / 10_000.0,
                    "area_percent": 100.0 * pixels / total if total else 0.0,
                    "threshold_method": f"{HISTOGRAM_BINS}-bin within-field histogram estimate",
                }
            )

    summary_path = args.index_dir / "vegetation_index_relative_classes.csv"
    with summary_path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    figure_path = args.index_dir / "vegetation_index_distribution.png"
    create_distribution_map(args.index_dir, args.boundary, figure_path)
    class_figure_path = args.index_dir / "vegetation_index_relative_classes.png"
    create_class_map(args.index_dir, args.boundary, class_figure_path)
    print(f"Wrote classified rasters, class summary, and maps to {args.index_dir}")
    for name in names:
        subset = [row for row in rows if row["index"] == name.upper()]
        thresholds = [row["upper_threshold"] for row in subset[:4]]
        print(f"{name.upper()} approximate 20/40/60/80% thresholds: {', '.join(thresholds)}")


if __name__ == "__main__":
    main()
