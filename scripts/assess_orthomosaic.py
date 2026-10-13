"""检查多光谱正射影像及研究区边界的质量。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import geometry_mask
from shapely.geometry import mapping


def scan(path: Path, boundary: Path) -> dict[str, object]:
    with rasterio.open(path) as dataset:
        if dataset.count != 5:
            raise ValueError(f"Expected four spectral bands plus alpha: {path}")
        boundary_gdf = gpd.read_file(boundary).to_crs(dataset.crs)
        geometries = [mapping(geometry) for geometry in boundary_gdf.geometry]
        band_names = [name or f"band_{index}" for index, name in
                      enumerate(dataset.descriptions[:4], 1)]
        result = {
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
                "sentinel_like_below_minus_9000": 0,
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
                summary["sentinel_like_below_minus_9000"] += int(
                    np.count_nonzero(values < -9000)
                )
                summary["nonfinite"] += int(np.count_nonzero(~np.isfinite(values)))
                finite = values[np.isfinite(values)]
                if finite.size:
                    summary["minimum"] = min(
                        summary["minimum"], float(finite.min())
                    )
                    summary["maximum"] = max(
                        summary["maximum"], float(finite.max())
                    )

        result["boundary_coverage_fraction"] = (
            result["boundary_alpha_valid_pixels"] / result["boundary_pixels"]
        )
        return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raster", type=Path, required=True)
    parser.add_argument("--boundary", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()

    result = scan(args.raster, args.boundary)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Wrote {args.output_json}")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
