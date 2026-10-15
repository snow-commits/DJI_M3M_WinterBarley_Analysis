"""Raw capture inventory and structural sanity checks.

The audit answers three questions before any processing happens: how many
capture groups exist, is every spectral band present, and do the bands inside a
group agree on size and data type?
"""

from __future__ import annotations

import csv
import warnings
from collections import defaultdict
from pathlib import Path
from time import perf_counter
from typing import Any

import rasterio
from rasterio.errors import NotGeoreferencedWarning

from m3m.constants import BANDS
from m3m.io import RGB_KEY, scan_capture_groups

EXPECTED_KEYS: tuple[str, ...] = (*BANDS, RGB_KEY)

INVENTORY_FIELDS: tuple[str, ...] = (
    "group_id",
    "rgb_exists",
    "g_exists",
    "r_exists",
    "re_exists",
    "nir_exists",
    "missing",
    "width",
    "height",
    "dtype",
    "metadata_issue",
)

__all__ = ["EXPECTED_KEYS", "INVENTORY_FIELDS", "audit_groups", "run", "summarize"]


def audit_groups(groups: dict[str, dict[str, Path]]) -> list[dict[str, Any]]:
    """Build one inventory row per capture group."""
    records: list[dict[str, Any]] = []

    for group_id in sorted(groups):
        files = groups[group_id]
        missing = [item for item in EXPECTED_KEYS if item not in files]

        row: dict[str, Any] = {
            "group_id": group_id,
            "rgb_exists": RGB_KEY in files,
            "g_exists": "G" in files,
            "r_exists": "R" in files,
            "re_exists": "RE" in files,
            "nir_exists": "NIR" in files,
            "missing": ",".join(missing),
            "width": "",
            "height": "",
            "dtype": "",
            "metadata_issue": "",
        }

        reference_shape: tuple[int, int] | None = None
        reference_dtype: str | None = None

        for band in BANDS:
            if band not in files:
                continue
            with rasterio.open(files[band]) as src:
                current_shape = (src.width, src.height)
                current_dtype = src.dtypes[0]
                if reference_shape is None:
                    reference_shape = current_shape
                    reference_dtype = current_dtype
                    row["width"] = src.width
                    row["height"] = src.height
                    row["dtype"] = current_dtype
                if current_shape != reference_shape:
                    row["metadata_issue"] += f"{band}_size_mismatch;"
                if current_dtype != reference_dtype:
                    row["metadata_issue"] += f"{band}_dtype_mismatch;"

        records.append(row)

    return records


def summarize(records: list[dict[str, Any]]) -> dict[str, int]:
    """Count complete and abnormal groups."""
    complete = sum(not row["missing"] and not row["metadata_issue"] for row in records)
    return {
        "groups": len(records),
        "complete": complete,
        "abnormal": len(records) - complete,
    }


def write_inventory(records: list[dict[str, Any]], path: Path | str) -> Path:
    """Write the inventory to a UTF-8-BOM CSV that Excel opens correctly."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(INVENTORY_FIELDS))
        writer.writeheader()
        writer.writerows(records)
    return path


def run(img_dir: Path | str, output_path: Path | str) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Scan *img_dir* and write the inventory CSV."""
    warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)
    start = perf_counter()

    groups = defaultdict(dict, scan_capture_groups(img_dir))
    records = audit_groups(groups)
    if not records:
        raise ValueError(f"No M3M captures found in {img_dir}")

    write_inventory(records, output_path)
    stats = summarize(records)
    stats["elapsed_seconds"] = round(perf_counter() - start, 3)
    return records, stats
