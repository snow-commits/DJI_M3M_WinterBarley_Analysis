"""在对齐网格上评估正射影像修复前后的质量。

修复后影像以最近邻方式重投影到修复前影像的网格，再对同一田界和像元统计；不修改源影像。
"""

from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.features import geometry_mask
from rasterio.vrt import WarpedVRT
from rasterio.windows import Window
from shapely.geometry import mapping


PROJECT = Path(r"E:\Projects\DJi\M3M_Project")
ORTHO = PROJECT / "outputs" / "orthomosaic"
BEFORE = ORTHO / "winterbarley_reflectance_orthomosaic_4band_qgis_fixed.tif"
AFTER = ORTHO / "winterbarley_reflectance_orthomosaic_4band_repaired.tif"
BOUNDARY = Path(
    r"E:\Projects\DJi\M3M_WinterBarley_Dataset"
    r"\M3M_WinterBarley_Dataset\boundary.gpkg"
)
OUTPUT_JSON = ORTHO / "repair_comparison_common_grid.json"
OUTPUT_FIGURE = ORTHO / "red_patch_before_after_common_grid_labeled.png"


def scan_common_grid() -> dict[str, object]:
    with rasterio.open(BEFORE) as before, rasterio.open(AFTER) as after:
        if before.crs != after.crs:
            raise ValueError(f"CRS mismatch: {before.crs} != {after.crs}")
        if before.count < 5 or after.count < 5:
            raise ValueError("Expected four spectral bands plus alpha in both rasters")

        boundary = gpd.read_file(BOUNDARY).to_crs(before.crs)
        shapes = [mapping(geom) for geom in boundary.geometry if geom is not None]
        band_names = [name or f"Band {i}" for i, name in enumerate(before.descriptions[:4], 1)]
        totals = {
            name: {
                "before_lt_0": 0, "before_gt_1": 0, "before_sentinel_lt_minus_9000": 0,
                "after_lt_0": 0, "after_gt_1": 0, "after_sentinel_lt_minus_9000": 0,
                "common_valid_pixels": 0, "changed_gt_1e-6": 0,
                "changed_gt_0_01": 0, "sum_abs_difference": 0.0,
                "sum_squared_difference": 0.0, "maximum_absolute_difference": 0.0,
                "paired_valid_reflectance_pixels": 0,
                "valid_changed_gt_0_01": 0,
                "valid_sum_abs_difference": 0.0,
                "valid_sum_squared_difference": 0.0,
            }
            for name in band_names
        }
        boundary_pixels = common_valid_pixels = 0

        # 最近邻重采样保留原始样本值，用于阈值检查。
        with WarpedVRT(
            after,
            crs=before.crs,
            transform=before.transform,
            width=before.width,
            height=before.height,
            resampling=Resampling.nearest,
        ) as aligned_after:
            for _, window in before.block_windows(1):
                old = before.read(window=window)
                new = aligned_after.read(window=window)
                inside = geometry_mask(
                    shapes,
                    out_shape=(int(window.height), int(window.width)),
                    transform=before.window_transform(window),
                    invert=True,
                )
                boundary_pixels += int(inside.sum())
                common_alpha = inside & (old[4] > 0) & (new[4] > 0)
                common_valid_pixels += int(common_alpha.sum())
                for band_index, name in enumerate(band_names):
                    a = old[band_index]
                    b = new[band_index]
                    valid = common_alpha & np.isfinite(a) & np.isfinite(b)
                    av = a[valid]
                    bv = b[valid]
                    stats = totals[name]
                    stats["common_valid_pixels"] += int(av.size)
                    stats["before_lt_0"] += int(np.count_nonzero(av < 0))
                    stats["before_gt_1"] += int(np.count_nonzero(av > 1))
                    stats["before_sentinel_lt_minus_9000"] += int(np.count_nonzero(av < -9000))
                    stats["after_lt_0"] += int(np.count_nonzero(bv < 0))
                    stats["after_gt_1"] += int(np.count_nonzero(bv > 1))
                    stats["after_sentinel_lt_minus_9000"] += int(np.count_nonzero(bv < -9000))
                    delta = bv.astype(np.float64) - av.astype(np.float64)
                    abs_delta = np.abs(delta)
                    stats["changed_gt_1e-6"] += int(np.count_nonzero(abs_delta > 1e-6))
                    stats["changed_gt_0_01"] += int(np.count_nonzero(abs_delta > 0.01))
                    stats["sum_abs_difference"] += float(abs_delta.sum(dtype=np.float64))
                    stats["sum_squared_difference"] += float(np.square(delta).sum(dtype=np.float64))
                    if abs_delta.size:
                        stats["maximum_absolute_difference"] = max(
                            stats["maximum_absolute_difference"], float(abs_delta.max())
                        )
                    # 排除无效或超范围的反射率，避免 -9999 占位值主导 RMSE。
                    reflectance_pair = (av >= 0) & (av <= 1) & (bv >= 0) & (bv <= 1)
                    valid_delta = delta[reflectance_pair]
                    stats["paired_valid_reflectance_pixels"] += int(valid_delta.size)
                    stats["valid_changed_gt_0_01"] += int(
                        np.count_nonzero(np.abs(valid_delta) > 0.01)
                    )
                    stats["valid_sum_abs_difference"] += float(
                        np.abs(valid_delta).sum(dtype=np.float64)
                    )
                    stats["valid_sum_squared_difference"] += float(
                        np.square(valid_delta).sum(dtype=np.float64)
                    )

        for stats in totals.values():
            count = stats["common_valid_pixels"]
            stats["mean_absolute_difference"] = (
                stats.pop("sum_abs_difference") / count if count else None
            )
            stats["rmse"] = (
                np.sqrt(stats.pop("sum_squared_difference") / count) if count else None
            )
            valid_count = stats["paired_valid_reflectance_pixels"]
            stats["valid_mean_absolute_difference"] = (
                stats.pop("valid_sum_abs_difference") / valid_count if valid_count else None
            )
            stats["valid_rmse"] = (
                np.sqrt(stats.pop("valid_sum_squared_difference") / valid_count)
                if valid_count else None
            )

        return {
            "method": "After raster nearest-neighbour aligned to before raster grid; statistics use identical boundary mask and common alpha-valid pixels.",
            "before_raster": str(BEFORE),
            "after_raster": str(AFTER),
            "crs": str(before.crs),
            "comparison_grid": {
                "width": before.width, "height": before.height,
                "pixel_size_x_m": abs(before.transform.a),
                "pixel_size_y_m": abs(before.transform.e),
                "transform": list(before.transform),
            },
            "boundary_pixels_on_comparison_grid": boundary_pixels,
            "common_alpha_valid_boundary_pixels": common_valid_pixels,
            "bands": totals,
        }


def make_patch_figure() -> None:
    with rasterio.open(BEFORE) as before, rasterio.open(AFTER) as after:
        # 保留报告中的红波段检查区域，并将两幅影像采样到完全相同的网格后绘图。
        window = Window(4450, 3450, 1000, 1000)
        bounds = before.window_bounds(window)
        extent = (bounds[0], bounds[2], bounds[1], bounds[3])
        old_red = before.read(1, window=window)
        with WarpedVRT(
            after,
            crs=before.crs,
            transform=before.transform,
            width=before.width,
            height=before.height,
            resampling=Resampling.nearest,
        ) as aligned_after:
            new_red = aligned_after.read(1, window=window)

    fig, axes = plt.subplots(1, 2, figsize=(11, 6.3), sharex=True, sharey=True)
    for ax, image, title in zip(axes, (old_red, new_red), ("Before repair", "After repair")):
        display = np.ma.masked_where(~np.isfinite(image) | (image < -9000), image)
        ax.imshow(
            display, cmap="gray", vmin=0, vmax=0.35, extent=extent,
            interpolation="nearest", origin="upper",
        )
        ax.set_title(title)
        ax.set_xlabel("Easting (m, UTM 33N)", labelpad=9)
        ax.ticklabel_format(useOffset=False, style="plain")
        ax.tick_params(axis="both", labelsize=9, pad=4)
    axes[0].set_ylabel("Northing (m, UTM 33N)", labelpad=9)
    fig.suptitle("Red-band reflectance: same geographic extent and 5 cm grid\nDisplay range 0–0.35; values above range appear white")
    # 为总标题和横轴标签预留空间，并保存完整图形边界，避免 Word 裁切底部标签。
    fig.tight_layout(rect=(0.04, 0.14, 0.99, 0.87))
    fig.savefig(OUTPUT_FIGURE, dpi=200, bbox_inches="tight", pad_inches=0.16)
    plt.close(fig)


def main() -> None:
    result = scan_common_grid()
    OUTPUT_JSON.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    make_patch_figure()
    print(f"Wrote {OUTPUT_JSON}")
    print(f"Wrote {OUTPUT_FIGURE}")
    print(json.dumps(result["bands"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
