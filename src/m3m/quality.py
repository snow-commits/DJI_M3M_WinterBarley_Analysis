"""Quality assessment of a four-band reflectance orthomosaic.

The check is intentionally conservative: it reports what is physically
impossible (negative reflectance, reflectance above 1, non-finite values,
NoData sentinels inside the field) and how much of the boundary the imagery
actually covers. Geometric accuracy is *not* assessed here - that needs
independently surveyed ground control points.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import geometry_mask
from shapely.geometry import mapping

from m3m.constants import NODATA

SENTINEL_THRESHOLD = NODATA + 900

__all__ = ["SENTINEL_THRESHOLD", "format_report", "run", "scan"]


def scan(path: Path | str, boundary: Path | str) -> dict[str, Any]:
    """Summarise band statistics inside the field boundary."""
    path = Path(path)
    with rasterio.open(path) as dataset:
        if dataset.count != 5:
            raise ValueError(f"Expected four spectral bands plus alpha: {path}")
        boundary_gdf = gpd.read_file(boundary).to_crs(dataset.crs)
        geometries = [mapping(geometry) for geometry in boundary_gdf.geometry]
        band_names = [
            name or f"band_{index}" for index, name in enumerate(dataset.descriptions[:4], 1)
        ]
        result: dict[str, Any] = {
            "raster": str(path),
            "crs": str(dataset.crs),
            "width": dataset.width,
            "height": dataset.height,
            "pixel_size_x_m": abs(dataset.transform.a),
            "pixel_size_y_m": abs(dataset.transform.e),
            "band_names": band_names,
            "data_types": list(dataset.dtypes),
            "nodata": dataset.nodata,
            "overviews": dataset.overviews(1),
            "boundary_area_m2": float(boundary_gdf.geometry.area.sum()),
            "boundary_pixels": 0,
            "boundary_alpha_valid_pixels": 0,
            "bands": {},
        }
        for band in band_names:
            result["bands"][band] = {
                "valid_pixels": 0,
                "less_than_zero": 0,
                "greater_than_one": 0,
                "sentinel_like": 0,
                "nonfinite": 0,
                "minimum": float("inf"),
                "maximum": float("-inf"),
            }

        for _, window in dataset.block_windows(1):
            pixels = dataset.read(window=window)
            inside = geometry_mask(
                geometries,
                out_shape=(int(window.height), int(window.width)),
                transform=dataset.window_transform(window),
                invert=True,
            )
            valid = inside & (pixels[4] > 0)
            result["boundary_pixels"] += int(inside.sum())
            result["boundary_alpha_valid_pixels"] += int(valid.sum())

            if not np.any(valid):
                continue
            for index, name in enumerate(band_names):
                values = pixels[index][valid]
                summary = result["bands"][name]
                summary["valid_pixels"] += int(values.size)
                summary["less_than_zero"] += int(np.count_nonzero(values < 0))
                summary["greater_than_one"] += int(np.count_nonzero(values > 1))
                summary["sentinel_like"] += int(
                    np.count_nonzero(values < SENTINEL_THRESHOLD)
                )
                summary["nonfinite"] += int(np.count_nonzero(~np.isfinite(values)))
                finite = values[np.isfinite(values)]
                if finite.size:
                    summary["minimum"] = min(summary["minimum"], float(finite.min()))
                    summary["maximum"] = max(summary["maximum"], float(finite.max()))

        boundary_pixels = result["boundary_pixels"]
        result["boundary_coverage_fraction"] = (
            result["boundary_alpha_valid_pixels"] / boundary_pixels if boundary_pixels else 0.0
        )
        return result


def format_report(result: dict[str, Any]) -> str:
    """Render the assessment as a short human-readable summary."""
    lines = [
        f"raster: {result['raster']}",
        f"  CRS {result['crs']}, {result['width']}x{result['height']} px, "
        f"pixel {result['pixel_size_x_m']:.4f} x {result['pixel_size_y_m']:.4f} m",
        f"  boundary coverage: {100 * result['boundary_coverage_fraction']:.2f}%",
    ]
    for name, summary in result["bands"].items():
        lines.append(
            f"  {name}: n={summary['valid_pixels']:,}, "
            f"range=[{summary['minimum']:.4f}, {summary['maximum']:.4f}], "
            f"<0={summary['less_than_zero']}, >1={summary['greater_than_one']}, "
            f"sentinel={summary['sentinel_like']}, non-finite={summary['nonfinite']}"
        )
    return "\n".join(lines)


def run(raster: Path | str, boundary: Path | str, output_json: Path | str) -> dict[str, Any]:
    """Assess *raster* and write the JSON report."""
    result = scan(raster, boundary)
    output_json = Path(output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return result
