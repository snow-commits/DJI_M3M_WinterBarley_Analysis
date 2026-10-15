"""Vegetation index rasters and memory-bounded statistics.

The whole orthomosaic is never loaded into memory: indices are computed window
by window and summarised into fixed-range histograms, so quantile estimates cost
a constant amount of RAM regardless of image size.
"""

from __future__ import annotations

import csv
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.features import geometry_mask
from shapely.geometry import mapping

from m3m.constants import (
    DEFAULT_ORTHOMOSAIC_BAND_MAP,
    EPSILON,
    HISTOGRAM_BINS,
    INDEX_SPECS,
    NODATA,
    ORTHOMOSAIC_BAND_KEYS,
    REQUIRED_ORTHOMOSAIC_BANDS,
)

#: Human-readable names used in raster band descriptions.
BAND_DESCRIPTIONS: dict[str, str] = {
    "red": "Red",
    "green": "Green",
    "blue": "Blue",
    "nir": "NIR",
    "rededge": "RedEdge",
    "alpha": "Alpha",
}

DEFAULT_INDEX_NAMES: tuple[str, ...] = tuple(INDEX_SPECS)

__all__ = [
    "BAND_DESCRIPTIONS",
    "DEFAULT_INDEX_NAMES",
    "IndexStats",
    "calculate",
    "format_summary",
    "resolve_band_positions",
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


def resolve_band_positions(
    band_map: Mapping[str, int] | None, band_count: int, path: Path | str
) -> dict[str, int]:
    """Validate a band-name -> position map against the raster's band count.

    Positions are 1-based; ``0`` marks a band the sensor does not have. Returns
    only the bands that are actually present.
    """
    band_map = dict(band_map or DEFAULT_ORTHOMOSAIC_BAND_MAP)
    unknown = {key for key in band_map if key not in ORTHOMOSAIC_BAND_KEYS}
    if unknown:
        raise ValueError(
            f"Unknown band name(s) {sorted(unknown)} in the band map. "
            f"Valid names: {', '.join(ORTHOMOSAIC_BAND_KEYS)}."
        )
    positions = {
        key: int(position) for key, position in band_map.items() if int(position) > 0
    }

    missing = [key for key in REQUIRED_ORTHOMOSAIC_BANDS if key not in positions]
    if missing:
        raise ValueError(
            f"The band map does not provide {', '.join(missing)}, which the indices "
            f"require. Configure [orthomosaic] band_map for this sensor."
        )
    out_of_range = {key: pos for key, pos in positions.items() if pos > band_count}
    if out_of_range:
        raise ValueError(
            f"The band map points past the end of the raster ({band_count} bands): "
            f"{out_of_range}. Check [orthomosaic] band_map."
        )
    seen: dict[int, str] = {}
    for key, position in sorted(positions.items()):
        if position in seen:
            raise ValueError(
                f"Bands {seen[position]!r} and {key!r} both map to position {position}."
            )
        seen[position] = key
    return positions


def validate_orthomosaic_bands(
    dataset: rasterio.io.DatasetReader,
    path: Path | str,
    positions: Mapping[str, int],
) -> None:
    """Cross-check the band map against the raster's own band descriptions.

    A wrong band map produces indices that look plausible but are nonsense, so
    when the export carries descriptions they are treated as authoritative. An
    export without descriptions is allowed through with no check.
    """
    descriptions = dataset.descriptions
    if not any(descriptions):
        return
    for key, position in sorted(positions.items()):
        expected = BAND_DESCRIPTIONS.get(key)
        actual = (descriptions[position - 1] or "").strip()
        if not actual or expected is None:
            continue
        if actual.lower() != expected.lower():
            raise ValueError(
                f"Band {position} of {path} is described as {actual!r}, but the band map "
                f"says {expected!r} is there. Fix [orthomosaic] band_map, or re-export "
                f"the orthomosaic with correct band descriptions."
            )


def calculate(
    raster_path: Path | str,
    boundary_path: Path | str,
    output_dir: Path | str,
    *,
    prefix: str = "field",
    index_names: Sequence[str] = DEFAULT_INDEX_NAMES,
    bins: int = HISTOGRAM_BINS,
    band_map: Mapping[str, int] | None = None,
) -> list[dict[str, Any]]:
    """Write one float32 raster per index plus a summary CSV.

    The mask is ``inside boundary AND alpha > 0 AND every required band finite
    and within [0, 1]``; the alpha clause is dropped for sensors that have no
    alpha band. Using the same mask for every index keeps the summary statistics
    directly comparable.
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
        positions = resolve_band_positions(band_map, src.count, raster_path)
        validate_orthomosaic_bands(src, raster_path, positions)
        # Deterministic read order; alpha is optional so it is handled separately.
        spectral = ("red", "nir", "rededge")
        band_mapping = ", ".join(
            f"{BAND_DESCRIPTIONS[key]}={positions[key]}"
            for key in ORTHOMOSAIC_BAND_KEYS
            if key in positions
        )
        mask_description = (
            "inside boundary AND alpha > 0 AND required input reflectance bands "
            "finite and within [0, 1]"
            if "alpha" in positions
            else "inside boundary AND required input reflectance bands finite and "
            "within [0, 1] (no alpha band configured)"
        )

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
                BAND_MAPPING=band_mapping,
                MASK=mask_description,
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
                read_order = spectral + (("alpha",) if "alpha" in positions else ())
                data = src.read(tuple(positions[key] for key in read_order), window=window)
                values = dict(zip(read_order, data, strict=True))
                red = values["red"]
                nir = values["nir"]
                rededge = values["rededge"]
                inside = geometry_mask(
                    geometries,
                    out_shape=(int(window.height), int(window.width)),
                    transform=src.window_transform(window),
                    invert=True,
                )
                valid = (
                    inside
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
                if "alpha" in values:
                    valid &= values["alpha"] > 0
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
