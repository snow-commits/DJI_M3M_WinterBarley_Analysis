import csv
import re
import warnings
from collections import defaultdict
from pathlib import Path
from time import perf_counter

import rasterio
from rasterio.errors import NotGeoreferencedWarning

from src.m3m.config import img_dir


warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)

EXPECTED = ["RGB", "G", "R", "RE", "NIR"]


def get_group_id(path):
    match = re.search(r"_(\d{4})(?:_MS_|_D$)", path.stem)

    if not match:
        return None

    return match.group(1)


def scan_dataset(directory):
    groups = defaultdict(dict)

    for path in directory.iterdir():
        if not path.is_file():
            continue

        stem_upper = path.stem.upper()
        suffix = path.suffix.lower()

        if suffix in [".tif", ".tiff"] and "_MS_" in stem_upper:
            band = stem_upper.split("_MS_")[-1]
            group_id = get_group_id(path)

            if group_id:
                groups[group_id][band] = path

        elif suffix in [".jpg", ".jpeg"] and stem_upper.endswith("_D"):
            group_id = get_group_id(path)

            if group_id:
                groups[group_id]["RGB"] = path

    return groups


def audit_groups(groups):
    records = []

    for group_id in sorted(groups):
        files = groups[group_id]
        missing = [item for item in EXPECTED if item not in files]

        row = {
            "group_id": group_id,
            "rgb_exists": "RGB" in files,
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

        reference_shape = None
        reference_dtype = None

        for band in ["G", "R", "RE", "NIR"]:
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
                    row["metadata_issue"] += f"{band}_尺寸异常;"

                if current_dtype != reference_dtype:
                    row["metadata_issue"] += f"{band}_数据类型异常;"

        records.append(row)

    return records


def main():
    start = perf_counter()

    groups = scan_dataset(img_dir)
    records = audit_groups(groups)

    output_dir = Path(__file__).resolve().parent / "outputs"
    output_dir.mkdir(exist_ok=True)

    output_path = output_dir / "full_dataset_inventory.csv"

    with output_path.open(
        mode="w",
        newline="",
        encoding="utf-8-sig"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=records[0].keys()
        )
        writer.writeheader()
        writer.writerows(records)

    complete_count = sum(
        not row["missing"] and not row["metadata_issue"]
        for row in records
    )

    abnormal_count = len(records) - complete_count
    elapsed = perf_counter() - start

    print("航拍组数量：", len(records))
    print("完整航拍组数量：", complete_count)
    print("缺失或异常组数量：", abnormal_count)
    print("结果文件：", output_path)
    print(f"耗时：{elapsed:.2f} 秒")


if __name__ == "__main__":
    main()