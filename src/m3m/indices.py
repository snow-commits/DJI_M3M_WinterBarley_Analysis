"""Vegetation index rasters and memory-bounded statistics.

The whole orthomosaic is never loaded into memory: indices are computed window
by window and summarised into fixed-range histograms, so quantile estimates cost
a constant amount of RAM regardless of image size.
"""

from __future__ import annotations

import csv
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.features import geometry_mask
from shapely.geometry import mapping

from m3m.constants import EPSILON, HISTOGRAM_BINS, INDEX_SPECS, NODATA

#: Band positions expected in a four-band reflectance orthomosaic plus alpha.
ORTHOMOSAIC_BANDS: dict[int, str] = {1: "Red", 3: "NIR", 4: "RedEdge"}
ALPHA_BAND = 5

DEFAULT_INDEX_NAMES: tuple[str, ...] = tuple(INDEX_SPECS)

__all__ = [
    "ALPHA_BAND",
    "DEFAULT_INDEX_NAMES",
    "ORTHOMOSAIC_BANDS",
    "IndexStats",
    "calculate",
    "format_summary",
    "validate_orthomosaic_bands",
]


@dataclass
class IndexStats:
    """Streaming accumulator: moments, extrema and a quantile histogram."""

    name: str
    value_range: tuple[float, float]
    bins: int = HISTOGRAM_BINS
    count: int = 0
    total: float = 0.0
    total_sq: float = 0.0
    minimum: float = float("inf")
    maximum: float = float("-inf")
    histogram: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.histogram = np.zeros(self.bins, dtype=np.int64)

    def add(self, values: np.ndarray) -> None:
        """Fold one window's valid values into the accumulator."""
        if values.size == 0:
            return
        values64 = values.astype(np.float64, copy=False)
        self.count += int(values64.size)
        self.total += float(values64.sum(dtype=np.float64))
        self.total_sq += float(np.square(values64).sum(dtype=np.float64))
        self.minimum = min(self.minimum, float(values64.min()))
        self.maximum = max(self.maximum, float(values64.max()))
        counts, _ = np.histogram(values64, bins=self.bins, range=self.value_range)
        self.histogram += counts

    @property
    def mean(self) -> float:
        return self.total / self.count if self.count else float("nan")

    @property
    def stddev(self) -> float:
        if not self.count:
            return float("nan")
        variance = max(self.total_sq / self.count - self.mean**2, 0.0)
        return float(np.sqrt(variance))

    def quantile(self, q: float) -> float:
        """Return a histogram-estimated quantile (accurate to the bin centre)."""
        if not self.count:
            return float("nan")
        target = q * (self.count - 1)
        cumulative = np.cumsum(self.histogram)
        bin_index = int(np.searchsorted(cumulative, target + 1, side="left"))
        low, high = self.value_range
        width = (high - low) / self.bins
        return float(low + (bin_index + 0.5) * width)


def validate_orthomosaic_bands(dataset: rasterio.io.DatasetReader, path: Path | str) -> None:
    """Check that band positions and descriptions match the expected layout."""
    if dataset.count < ALPHA_BAND:
        raise ValueError(
            f"Expected four reflectance bands plus an alpha band, got {dataset.count}: {path}"
        )
    descriptions = dataset.descriptions
    for band, expected in ORTHOMOSAIC_BANDS.items():
        actual = (descriptions[band - 1] or "").lower()
        if actual != expected.lower():
            raise ValueError(
                f"Band {band} of {path} should be {expected!r} but its description is "
                f"{actual!r}. Re-export the orthomosaic with band descriptions set."
            )


def calculate(
    raster_path: Path | str,
    boundary_path: Path | str,
    output_dir: Path | str,
    *,
    prefix: str = "field",
    index_names: Sequence[str] = DEFAULT_INDEX_NAMES,
    bins: int = HISTOGRAM_BINS,
) -> list[dict[str, Any]]:
    """Write one float32 raster per index plus a summary CSV.

    The mask is ``inside boundary AND alpha > 0 AND every required band finite
    and within [0, 1]``. Using the same mask for every index keeps the summary
    statistics directly comparable.
    """
    raster_path = Path(raster_path)
    boundary_path = Path(boundary_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    unknown = set(index_names) - set(INDEX_SPECS)
    if unknown:
        raise ValueError(
            f"Unknown index name(s): {', '.join(sorted(unknown))}. "
            f"Available: {', '.join(INDEX_SPECS)}"
        )

    with rasterio.open(raster_path) as src:
        validate_orthomosaic_bands(src, raster_path)

        boundary = gpd.read_file(boundary_path)
        if boundary.empty or boundary.geometry.is_empty.all():
            raise ValueError(f"No usable boundary geometry in {boundary_path}")
        boundary = boundary.to_crs(src.crs)
        geometries = [
            mapping(geom)
            for geom in boundary.geometry
            if geom is not None and not geom.is_empty
        ]
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
        destinations: dict[str, rasterio.io.DatasetWriter] = {}
        for name in index_names:
            path = output_dir / f"{prefix}_{name.lower()}.tif"
            dst = rasterio.open(path, "w", **profile)
            dst.set_band_description(1, name)
            dst.update_tags(
                1,
                INDEX_NAME=name,
                FORMULA=INDEX_SPECS[name]["formula"],
                INPUT_RASTER=str(raster_path),
                BAND_MAPPING="Red=1, NIR=3, RedEdge=4, Alpha=5",
                MASK=(
                    "inside boundary AND alpha > 0 AND required input reflectance "
                    "bands finite and within [0, 1]"
                ),
                EVI2_REFLECTANCE_SCALE="Input reflectance is unitless, scaled 0-1",
            )
            destinations[name] = dst

        stats = {
            name: IndexStats(
                name=name,
                value_range=INDEX_SPECS[name]["histogram_range"],  # type: ignore[arg-type]
                bins=bins,
            )
            for name in index_names
        }
        try:
            for _, window in src.block_windows(1):
                data = src.read((1, 3, 4, ALPHA_BAND), window=window)
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

                computed = {
                    "NDVI": (
                        (nir - red) / np.where(ndvi_denominator > EPSILON, ndvi_denominator, 1),
                        valid & (ndvi_denominator > EPSILON),
                    ),
                    "NDRE": (
                        (nir - rededge)
                        / np.where(ndre_denominator > EPSILON, ndre_denominator, 1),
                        valid & (ndre_denominator > EPSILON),
                    ),
                    "EVI2": (
                        2.5
                        * (nir - red)
                        / np.where(evi2_denominator > EPSILON, evi2_denominator, 1),
                        valid & (evi2_denominator > EPSILON),
                    ),
                }
                for name in index_names:
                    values, local_valid = computed[name]
                    out = np.full(red.shape, NODATA, dtype=np.float32)
                    finite = local_valid & np.isfinite(values)
                    out[finite] = values[finite].astype(np.float32, copy=False)
                    destinations[name].write(out, 1, window=window)
                    stats[name].add(out[finite])
        finally:
            for dst in destinations.values():
                dst.close()

    rows: list[dict[str, Any]] = []
    for name in index_names:
        stat = stats[name]
        if stat.count == 0:
            raise RuntimeError(f"No valid pixels were calculated for {name}.")
        rows.append(
            {
                "index": name,
                "formula": INDEX_SPECS[name]["formula"],
                "valid_pixels": stat.count,
                "mean": stat.mean,
                "stddev": stat.stddev,
                "median_histogram_estimate": stat.quantile(0.5),
                "p95_histogram_estimate": stat.quantile(0.95),
                "minimum": stat.minimum,
                "maximum": stat.maximum,
                "boundary_area_ha": boundary_area_m2 / 10_000.0,
                "quantile_method": f"{bins}-bin histogram approximation",
            }
        )
        out_path = output_dir / f"{prefix}_{name.lower()}.tif"
        with rasterio.open(out_path, "r+") as dst:
            dst.update_tags(
                1,
                STATISTICS_MINIMUM=str(stat.minimum),
                STATISTICS_MAXIMUM=str(stat.maximum),
                STATISTICS_MEAN=str(stat.mean),
                STATISTICS_STDDEV=str(stat.stddev),
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


def format_summary(rows: Iterable[dict[str, Any]]) -> str:
    """Render summary rows as one line per index."""
    lines = []
    for row in rows:
        lines.append(
            f"{row['index']}: n={row['valid_pixels']:,}, "
            f"mean={row['mean']:.6f}, median~{row['median_histogram_estimate']:.6f}, "
            f"P95~{row['p95_histogram_estimate']:.6f}, "
            f"range=[{row['minimum']:.6f}, {row['maximum']:.6f}]"
        )
    return "\n".join(lines)
