"""在共同网格上比较指数值及田内五分位分区。"""

from __future__ import annotations

import argparse
import csv
from itertools import combinations
from pathlib import Path

import numpy as np
import rasterio


INDEX_NAMES = ("ndvi", "ndre", "evi2")


def compare(index_dir: Path) -> list[dict[str, object]]:
    paths = {
        name: index_dir / f"winterbarley_{name}.tif" for name in INDEX_NAMES
    }
    class_paths = {
        name: index_dir / f"winterbarley_{name}_relative_5class.tif"
        for name in INDEX_NAMES
    }
    rows: list[dict[str, object]] = []
    with rasterio.open(paths[INDEX_NAMES[0]]) as ref:
        for path in list(paths.values())[1:] + list(class_paths.values()):
            with rasterio.open(path) as ds:
                if (ds.width, ds.height, ds.transform, ds.crs) != (
                    ref.width,
                    ref.height,
                    ref.transform,
                    ref.crs,
                ):
                    raise ValueError(f"Raster grid does not match: {path}")

        for name_a, name_b in combinations(INDEX_NAMES, 2):
            sum_x = sum_y = sum_x2 = sum_y2 = sum_xy = 0.0
            count = agreement = 0
            low_intersection = low_union = 0
            high_intersection = high_union = 0
            with rasterio.open(paths[name_a]) as a_ds, rasterio.open(paths[name_b]) as b_ds, rasterio.open(class_paths[name_a]) as a_class_ds, rasterio.open(class_paths[name_b]) as b_class_ds:
                for _, window in ref.block_windows(1):
                    a = a_ds.read(1, window=window)
                    b = b_ds.read(1, window=window)
                    valid = np.isfinite(a) & np.isfinite(b)
                    if a_ds.nodata is not None:
                        valid &= a != a_ds.nodata
                    if b_ds.nodata is not None:
                        valid &= b != b_ds.nodata
                    if not np.any(valid):
                        continue

                    x = a[valid].astype(np.float64, copy=False)
                    y = b[valid].astype(np.float64, copy=False)
                    sum_x += float(x.sum(dtype=np.float64))
                    sum_y += float(y.sum(dtype=np.float64))
                    sum_x2 += float(np.square(x).sum(dtype=np.float64))
                    sum_y2 += float(np.square(y).sum(dtype=np.float64))
                    sum_xy += float(np.multiply(x, y).sum(dtype=np.float64))
                    count += int(x.size)

                    ca = a_class_ds.read(1, window=window)[valid]
                    cb = b_class_ds.read(1, window=window)[valid]
                    agreement += int(np.count_nonzero(ca == cb))
                    low_a, low_b = ca == 1, cb == 1
                    high_a, high_b = ca == 5, cb == 5
                    low_intersection += int(np.count_nonzero(low_a & low_b))
                    low_union += int(np.count_nonzero(low_a | low_b))
                    high_intersection += int(np.count_nonzero(high_a & high_b))
                    high_union += int(np.count_nonzero(high_a | high_b))

            if count == 0:
                raise RuntimeError(f"No common valid pixels for {name_a} and {name_b}")
            cov_numerator = sum_xy - sum_x * sum_y / count
            var_x = max(sum_x2 - sum_x * sum_x / count, 0.0)
            var_y = max(sum_y2 - sum_y * sum_y / count, 0.0)
            pearson = cov_numerator / np.sqrt(var_x * var_y)
            rows.append(
                {
                    "index_pair": f"{name_a.upper()} vs {name_b.upper()}",
                    "common_valid_pixels": count,
                    "pearson_r": float(pearson),
                    "same_quintile_class_percent": 100.0 * agreement / count,
                    "lowest_class_intersection_pixels": low_intersection,
                    "lowest_class_jaccard": low_intersection / low_union if low_union else float("nan"),
                    "highest_class_intersection_pixels": high_intersection,
                    "highest_class_jaccard": high_intersection / high_union if high_union else float("nan"),
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index-dir", type=Path, required=True)
    args = parser.parse_args()
    rows = compare(args.index_dir)
    path = args.index_dir / "vegetation_index_pairwise_comparison.csv"
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {path}")
    for row in rows:
        print(
            f"{row['index_pair']}: Pearson r={row['pearson_r']:.4f}, "
            f"same class={row['same_quintile_class_percent']:.2f}%, "
            f"lowest/highest Jaccard={row['lowest_class_jaccard']:.4f}/"
            f"{row['highest_class_jaccard']:.4f}"
        )


if __name__ == "__main__":
    main()
