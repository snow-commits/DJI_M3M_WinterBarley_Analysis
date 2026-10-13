"""仅填补 ODM 工作副本中少量辐射 NoData 像元。

ODM 浮点纹理融合可能将饱和像元的 -9999 占位值扩散成亮暗异常块。源反射率影像保持不变；
本脚本应处理带元数据的 ODM 副本。
"""

from __future__ import annotations

import argparse
import csv
import ntpath
from pathlib import Path

import numpy as np
import rasterio
from scipy.ndimage import distance_transform_edt


def image_name(source: str) -> str:
    filename = ntpath.basename(source)
    if not filename.upper().endswith(".TIF") or "_MS_" not in filename:
        raise ValueError(f"Unexpected source image name: {source}")
    return filename[:-4].replace("_MS_", "_") + ".tif"


def fill_image(path: Path, dry_run: bool) -> dict[str, object]:
    mode = "r" if dry_run else "r+"
    with rasterio.open(path, mode) as dataset:
        if dataset.count != 1 or dataset.dtypes[0] != "float32":
            raise ValueError(f"Expected a single float32 band: {path}")

        values = dataset.read(1)
        invalid = ~np.isfinite(values) | (values <= -9000)
        invalid_count = int(invalid.sum())
        if np.any((values < 0) & ~invalid):
            raise ValueError(f"Unexpected negative reflectance values: {path}")
        if not np.any(~invalid):
            raise ValueError(f"No usable pixels: {path}")

        max_distance = 0.0
        if invalid_count:
            distance, nearest = distance_transform_edt(
                invalid, return_indices=True
            )
            max_distance = float(distance[invalid].max())
            values[invalid] = values[tuple(nearest[:, invalid])]

            if not np.all(np.isfinite(values)) or np.any(values < 0):
                raise ValueError(f"Filling left invalid values: {path}")
            if not dry_run:
                dataset.write(values, 1)
                dataset.update_tags(
                    ODM_FILL_METHOD="nearest valid pixel",
                    ODM_FILLED_PIXEL_COUNT=str(invalid_count),
                )

        return {
            "image": path.name,
            "invalid_pixels": invalid_count,
            "maximum_fill_distance_pixels": f"{max_distance:.3f}",
            "maximum_reflectance": f"{float(values.max()):.6f}",
            "status": "dry_run" if dry_run else ("filled" if invalid_count else "clean"),
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--audit-csv", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    with args.manifest.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    selected = [
        row for row in rows
        if row["status"] == "ok" and float(row["valid_fraction"]) < 1
    ]
    if args.limit is not None:
        selected = selected[: args.limit]

    results = []
    for index, row in enumerate(selected, 1):
        path = args.images_dir / image_name(row["source"])
        if not path.is_file():
            raise FileNotFoundError(path)
        result = fill_image(path, args.dry_run)
        result["band"] = row["band"]
        result["group_id"] = row["group_id"]
        results.append(result)
        if index % 25 == 0 or index == len(selected):
            print(f"Checked {index}/{len(selected)} images", flush=True)

    args.audit_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.audit_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "group_id", "band", "image", "invalid_pixels",
                "maximum_fill_distance_pixels", "maximum_reflectance", "status",
            ],
        )
        writer.writeheader()
        writer.writerows(results)

    print(
        f"Images checked: {len(results)}; pixels filled: "
        f"{sum(int(item['invalid_pixels']) for item in results)}; "
        f"audit: {args.audit_csv}"
    )


if __name__ == "__main__":
    main()
