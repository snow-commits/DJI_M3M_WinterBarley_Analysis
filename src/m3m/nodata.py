"""Fill saturated NoData pixels in the ODM working copy.

OpenDroneMap's float texture blending can smear the ``-9999`` saturation
placeholder into bright or dark blocks in the orthomosaic. The fix is applied
only to the throw-away ODM working copy: source reflectance images are never
modified.

Each filled pixel is replaced by its nearest valid neighbour and the operation
is recorded in an audit CSV, including the maximum fill distance, so the amount
of invention can be reviewed rather than assumed.
"""

from __future__ import annotations

import csv
import ntpath
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from scipy.ndimage import distance_transform_edt

from m3m.constants import NODATA

FILL_AUDIT_FIELDS: tuple[str, ...] = (
    "group_id",
    "band",
    "image",
    "invalid_pixels",
    "maximum_fill_distance_pixels",
    "maximum_reflectance",
    "status",
)

__all__ = ["FILL_AUDIT_FIELDS", "fill_image", "image_name", "run", "select_targets"]


def image_name(source: str) -> str:
    """Map a calibration manifest source path to its ODM working-copy name."""
    filename = ntpath.basename(source)
    if not filename.upper().endswith(".TIF") or "_MS_" not in filename:
        raise ValueError(f"Unexpected source image name: {source}")
    return filename[:-4].replace("_MS_", "_") + ".tif"


def fill_image(path: Path | str, dry_run: bool = False) -> dict[str, Any]:
    """Replace invalid pixels of one float32 band with their nearest valid neighbour."""
    path = Path(path)
    mode = "r" if dry_run else "r+"
    with rasterio.open(path, mode) as dataset:
        if dataset.count != 1 or dataset.dtypes[0] != "float32":
            raise ValueError(f"Expected a single float32 band: {path}")

        values = dataset.read(1)
        invalid = ~np.isfinite(values) | (values <= NODATA + 900)
        invalid_count = int(invalid.sum())
        if np.any((values < 0) & ~invalid):
            raise ValueError(f"Unexpected negative reflectance values: {path}")
        if not np.any(~invalid):
            raise ValueError(f"No usable pixels: {path}")

        max_distance = 0.0
        if invalid_count:
            distance, nearest = distance_transform_edt(invalid, return_indices=True)
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


def select_targets(rows: list[dict[str, str]], limit: int | None = None) -> list[dict[str, str]]:
    """Pick manifest rows that succeeded but still contain saturated pixels."""
    selected = [
        row for row in rows if row["status"] == "ok" and float(row["valid_fraction"]) < 1
    ]
    return selected[:limit] if limit is not None else selected


def run(
    images_dir: Path | str,
    manifest: Path | str,
    audit_csv: Path | str,
    *,
    dry_run: bool = False,
    limit: int | None = None,
    progress_every: int = 25,
) -> list[dict[str, Any]]:
    """Fill every affected image and write the fill audit CSV."""
    images_dir = Path(images_dir)
    with Path(manifest).open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    selected = select_targets(rows, limit)

    results: list[dict[str, Any]] = []
    for index, row in enumerate(selected, 1):
        path = images_dir / image_name(row["source"])
        if not path.is_file():
            raise FileNotFoundError(path)
        result = fill_image(path, dry_run)
        result["band"] = row["band"]
        result["group_id"] = row["group_id"]
        results.append(result)
        if progress_every and (index % progress_every == 0 or index == len(selected)):
            print(f"Checked {index}/{len(selected)} images", flush=True)

    audit_csv = Path(audit_csv)
    audit_csv.parent.mkdir(parents=True, exist_ok=True)
    with audit_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(FILL_AUDIT_FIELDS))
        writer.writeheader()
        writer.writerows(results)

    print(
        f"Images checked: {len(results)}; pixels filled: "
        f"{sum(int(item['invalid_pixels']) for item in results)}; audit: {audit_csv}"
    )
    return results
