"""根据修复后的 M3M 正射影像计算 NDVI、NDRE 和 EVI2。"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.features import geometry_mask
from shapely.geometry import mapping


INDEX_SPECS = {
    "NDVI": {
        "formula": "(NIR - Red) / (NIR + Red)",
        "histogram_range": (-1.0, 1.0),
    },
    "NDRE": {
        "formula": "(NIR - RedEdge) / (NIR + RedEdge)",
        "histogram_range": (-1.0, 1.0),
    },
    "EVI2": {
        "formula": "2.5 * (NIR - Red) / (NIR + 2.4 * Red + 1)",
        "histogram_range": (-2.5 / 3.4, 1.25),
    },
}
HISTOGRAM_BINS = 100_000
NODATA = -9999.0
EPSILON = 1e-8


@dataclass
class IndexStats:
    count: int
    total: float
    total_sq: float
    minimum: float
    maximum: float
    histogram: np.ndarray

    @classmethod
    def create(cls, name: str) -> "IndexStats":
        return cls(
            count=0,
            total=0.0,
            total_sq=0.0,
            minimum=float("inf"),
            maximum=float("-inf"),
            histogram=np.zeros(HISTOGRAM_BINS, dtype=np.int64),
        )

    def add(self, values: np.ndarray, name: str) -> None:
        if values.size == 0:
            return
        values64 = values.astype(np.float64, copy=False)
        self.count += int(values.size)
        self.total += float(values64.sum(dtype=np.float64))
        self.total_sq += float(np.square(values64).sum(dtype=np.float64))
        self.minimum = min(self.minimum, float(values64.min()))
        self.maximum = max(self.maximum, float(values64.max()))
        low, high = INDEX_SPECS[name]["histogram_range"]
        counts, _ = np.histogram(values64, bins=HISTOGRAM_BINS, range=(low, high))
        self.histogram += counts

    def quantile(self, q: float, name: str) -> float:
        """返回按直方图估算的分位数（精度为箱中心）。"""
        if not self.count:
            return float("nan")
        target = q * (self.count - 1)
        cumulative = np.cumsum(self.histogram)
        bin_index = int(np.searchsorted(cumulative, target + 1, side="left"))
        low, high = INDEX_SPECS[name]["histogram_range"]
        width = (high - low) / HISTOGRAM_BINS
        return float(low + (bin_index + 0.5) * width)


def calculate(raster_path: Path, boundary_path: Path, output_dir: Path) -> list[dict[str, object]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    with rasterio.open(raster_path) as src:
        if src.count < 5:
            raise ValueError(f"Expected four reflectance bands and alpha band, got {src.count}.")
        descriptions = src.descriptions
        expected = {1: "Red", 3: "NIR", 4: "RedEdge"}
        for band, name in expected.items():
            actual = (descriptions[band - 1] or "").lower()
            if actual != name.lower():
                raise ValueError(f"Band {band} should be {name}, but metadata says {actual!r}.")

        boundary = gpd.read_file(boundary_path)
        if boundary.empty or boundary.geometry.is_empty.all():
            raise ValueError(f"No usable boundary geometry in {boundary_path}")
        boundary = boundary.to_crs(src.crs)
        geometries = [mapping(g) for g in boundary.geometry if g is not None and not g.is_empty]
        boundary_area_m2 = float(boundary.geometry.area.sum())

        profile = src.profile.copy()
        profile.update(
            count=1,
            dtype="float32",
            nodata=NODATA,
            compress="deflate",
            predictor=3,
            tiled=True,
            blockxsize=512,
            blockysize=512,
            BIGTIFF="IF_SAFER",
        )
        destinations = {}
        for name, spec in INDEX_SPECS.items():
            path = output_dir / f"winterbarley_{name.lower()}.tif"
            dst = rasterio.open(path, "w", **profile)
            dst.set_band_description(1, name)
            dst.update_tags(
                1,
                INDEX_NAME=name,
                FORMULA=spec["formula"],
                INPUT_RASTER=str(raster_path),
                BAND_MAPPING="Red=1, NIR=3, RedEdge=4, Alpha=5",
                MASK="inside boundary.gpkg AND alpha > 0 AND required input reflectance bands finite in [0,1]",
                EVI2_REFLECTANCE_SCALE="Input reflectance is unitless, scaled 0-1",
            )
            destinations[name] = dst

        stats = {name: IndexStats.create(name) for name in INDEX_SPECS}
        try:
            for _, window in src.block_windows(1):
                data = src.read((1, 3, 4, 5), window=window)
                red, nir, rededge, alpha = data
                inside = geometry_mask(
                    geometries,
                    out_shape=(int(window.height), int(window.width)),
                    transform=src.window_transform(window),
                    invert=True,
                )
                valid = (
                    inside
                    & (alpha > 0)
                    & np.isfinite(red)
                    & np.isfinite(nir)
                    & np.isfinite(rededge)
                    & (red >= 0)
                    & (red <= 1)
                    & (nir >= 0)
                    & (nir <= 1)
                    & (rededge >= 0)
                    & (rededge <= 1)
                )
                ndvi_denominator = nir + red
                ndre_denominator = nir + rededge
                evi2_denominator = nir + 2.4 * red + 1.0
                ndvi_valid = valid & (ndvi_denominator > EPSILON)
                ndre_valid = valid & (ndre_denominator > EPSILON)
                evi2_valid = valid & (evi2_denominator > EPSILON)

                for name, values, local_valid in (
                    ("NDVI", (nir - red) / np.where(ndvi_denominator > EPSILON, ndvi_denominator, 1), ndvi_valid),
                    ("NDRE", (nir - rededge) / np.where(ndre_denominator > EPSILON, ndre_denominator, 1), ndre_valid),
                    ("EVI2", 2.5 * (nir - red) / np.where(evi2_denominator > EPSILON, evi2_denominator, 1), evi2_valid),
                ):
                    out = np.full(red.shape, NODATA, dtype=np.float32)
                    finite = local_valid & np.isfinite(values)
                    out[finite] = values[finite].astype(np.float32, copy=False)
                    destinations[name].write(out, 1, window=window)
                    stats[name].add(out[finite], name)
        finally:
            for dst in destinations.values():
                dst.close()

    rows: list[dict[str, object]] = []
    for name, stat in stats.items():
        if stat.count == 0:
            raise RuntimeError(f"No valid pixels were calculated for {name}.")
        mean = stat.total / stat.count
        variance = max(stat.total_sq / stat.count - mean * mean, 0.0)
        row = {
            "index": name,
            "formula": INDEX_SPECS[name]["formula"],
            "valid_pixels": stat.count,
            "mean": mean,
            "stddev": float(np.sqrt(variance)),
            "median_histogram_estimate": stat.quantile(0.5, name),
            "p95_histogram_estimate": stat.quantile(0.95, name),
            "minimum": stat.minimum,
            "maximum": stat.maximum,
            "boundary_area_ha": boundary_area_m2 / 10_000.0,
            "quantile_method": f"{HISTOGRAM_BINS}-bin histogram approximation",
        }
        rows.append(row)
        out_path = output_dir / f"winterbarley_{name.lower()}.tif"
        with rasterio.open(out_path, "r+") as dst:
            dst.update_tags(
                1,
                STATISTICS_MINIMUM=str(stat.minimum),
                STATISTICS_MAXIMUM=str(stat.maximum),
                STATISTICS_MEAN=str(mean),
                STATISTICS_STDDEV=str(np.sqrt(variance)),
                STATISTICS_VALID_PIXELS=str(stat.count),
            )
            dst.build_overviews([2, 4, 8, 16], Resampling.average)
            dst.update_tags(ns="rio_overview", resampling="average")

    summary_path = output_dir / "vegetation_index_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raster", type=Path, required=True)
    parser.add_argument("--boundary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    rows = calculate(args.raster, args.boundary, args.output_dir)
    print(f"Wrote index rasters and summary to {args.output_dir}")
    for row in rows:
        print(
            f"{row['index']}: n={row['valid_pixels']:,}, "
            f"mean={row['mean']:.6f}, median≈{row['median_histogram_estimate']:.6f}, "
            f"P95≈{row['p95_histogram_estimate']:.6f}, "
            f"range=[{row['minimum']:.6f}, {row['maximum']:.6f}]"
        )


if __name__ == "__main__":
    main()
