"""Pairwise comparisons between rasters.

Two independent analyses live here, both purely descriptive:

* :func:`compare_indices` - how much do NDVI, NDRE and EVI2 agree with each
  other, in value and in relative class? High agreement means the choice of
  index barely matters for zoning; it says nothing about which index is *right*,
  because no ground truth enters the comparison.
* :func:`compare_repairs` - what did the orthomosaic repair actually change?
  The repaired raster is resampled onto the pre-repair grid with nearest
  neighbour so both share an identical mask and grid.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from itertools import combinations
from pathlib import Path
from typing import Any

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import rasterio  # noqa: E402
from rasterio.enums import Resampling  # noqa: E402
from rasterio.features import geometry_mask  # noqa: E402
from rasterio.vrt import WarpedVRT  # noqa: E402
from rasterio.windows import Window  # noqa: E402
from shapely.geometry import mapping  # noqa: E402

from m3m.constants import INDEX_ORDER, NODATA  # noqa: E402
from m3m.zones import class_raster_path, require_index_rasters  # noqa: E402

SENTINEL_THRESHOLD = NODATA + 900

__all__ = [
    "INDEX_COMPARISON_FIELDS",
    "compare_indices",
    "compare_repairs",
    "default_patch_window",
]


INDEX_COMPARISON_FIELDS = (
    "index_pair",
    "common_valid_pixels",
    "pearson_r",
    "same_quintile_class_percent",
    "lowest_class_intersection_pixels",
    "lowest_class_jaccard",
    "highest_class_intersection_pixels",
    "highest_class_jaccard",
)


def compare_indices(
    index_dir: Path | str,
    *,
    prefix: str = "field",
    index_names: Sequence[str] = INDEX_ORDER,
) -> list[dict[str, Any]]:
    """Compare value agreement and relative-class agreement for every index pair."""
    index_dir = Path(index_dir)
    paths = require_index_rasters(index_dir, prefix, index_names)
    class_paths = {name: class_raster_path(index_dir, prefix, name) for name in index_names}

    rows: list[dict[str, Any]] = []
    with rasterio.open(paths[index_names[0]]) as reference:
        for path in list(paths.values())[1:] + list(class_paths.values()):
            with rasterio.open(path) as dataset:
                if (dataset.width, dataset.height, dataset.transform, dataset.crs) != (
                    reference.width,
                    reference.height,
                    reference.transform,
                    reference.crs,
                ):
                    raise ValueError(f"Raster grid does not match: {path}")

        for name_a, name_b in combinations(index_names, 2):
            sum_x = sum_y = sum_x2 = sum_y2 = sum_xy = 0.0
            count = agreement = 0
            low_intersection = low_union = 0
            high_intersection = high_union = 0
            with (
                rasterio.open(paths[name_a]) as a_ds,
                rasterio.open(paths[name_b]) as b_ds,
                rasterio.open(class_paths[name_a]) as a_class_ds,
                rasterio.open(class_paths[name_b]) as b_class_ds,
            ):
                for _, window in reference.block_windows(1):
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

                    classes_a = a_class_ds.read(1, window=window)[valid]
                    classes_b = b_class_ds.read(1, window=window)[valid]
                    agreement += int(np.count_nonzero(classes_a == classes_b))
                    low_a, low_b = classes_a == 1, classes_b == 1
                    high_a, high_b = classes_a == 5, classes_b == 5
                    low_intersection += int(np.count_nonzero(low_a & low_b))
                    low_union += int(np.count_nonzero(low_a | low_b))
                    high_intersection += int(np.count_nonzero(high_a & high_b))
                    high_union += int(np.count_nonzero(high_a | high_b))

            if count == 0:
                raise RuntimeError(f"No common valid pixels for {name_a} and {name_b}")
            covariance = sum_xy - sum_x * sum_y / count
            variance_x = max(sum_x2 - sum_x * sum_x / count, 0.0)
            variance_y = max(sum_y2 - sum_y * sum_y / count, 0.0)
            pearson = covariance / np.sqrt(variance_x * variance_y)
            rows.append(
                {
                    "index_pair": f"{name_a.upper()} vs {name_b.upper()}",
                    "common_valid_pixels": count,
                    "pearson_r": float(pearson),
                    "same_quintile_class_percent": 100.0 * agreement / count,
                    "lowest_class_intersection_pixels": low_intersection,
                    "lowest_class_jaccard": (
                        low_intersection / low_union if low_union else float("nan")
                    ),
                    "highest_class_intersection_pixels": high_intersection,
                    "highest_class_jaccard": (
                        high_intersection / high_union if high_union else float("nan")
                    ),
                }
            )
    return rows


def default_patch_window(reference: rasterio.io.DatasetReader, size: int = 1000) -> Window:
    """A centred window used for the before/after visual check."""
    width = min(size, reference.width)
    height = min(size, reference.height)
    column = max(0, (reference.width - width) // 2)
    row = max(0, (reference.height - height) // 2)
    return Window(column, row, width, height)


def compare_repairs(
    before: Path | str,
    after: Path | str,
    boundary: Path | str,
    *,
    output_json: Path | str | None = None,
    output_figure: Path | str | None = None,
    patch_window: tuple[int, int, int, int] | None = None,
    figure_band: int = 1,
    display_range: tuple[float, float] = (0.0, 0.35),
) -> dict[str, Any]:
    """Quantify and visualise what the orthomosaic repair changed.

    ``patch_window`` is ``(column, row, width, height)`` in *before* raster
    pixels. When omitted a centred window is used.
    """
    before = Path(before)
    after = Path(after)
    statistics = _scan_common_grid(before, after, boundary)

    if patch_window is None:
        with rasterio.open(before) as reference:
            window = default_patch_window(reference)
    else:
        window = Window(*patch_window)

    if output_figure is not None:
        _make_patch_figure(before, after, Path(output_figure), window, figure_band, display_range)
        statistics["figure"] = str(output_figure)
    if output_json is not None:
        Path(output_json).parent.mkdir(parents=True, exist_ok=True)
        Path(output_json).write_text(
            json.dumps(statistics, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        statistics["report"] = str(output_json)
    return statistics


def _scan_common_grid(before: Path, after: Path, boundary: Path) -> dict[str, Any]:
    with rasterio.open(before) as before_ds, rasterio.open(after) as after_ds:
        if before_ds.crs != after_ds.crs:
            raise ValueError(f"CRS mismatch: {before_ds.crs} != {after_ds.crs}")
        if before_ds.count < 5 or after_ds.count < 5:
            raise ValueError("Expected four spectral bands plus alpha in both rasters")

        boundary_gdf = gpd.read_file(boundary).to_crs(before_ds.crs)
        shapes = [mapping(geom) for geom in boundary_gdf.geometry if geom is not None]
        band_names = [
            name or f"Band {index}"
            for index, name in enumerate(before_ds.descriptions[:4], 1)
        ]
        totals: dict[str, dict[str, Any]] = {
            name: {
                "before_lt_0": 0,
                "before_gt_1": 0,
                "before_sentinel_like": 0,
                "after_lt_0": 0,
                "after_gt_1": 0,
                "after_sentinel_like": 0,
                "common_valid_pixels": 0,
                "changed_gt_1e-6": 0,
                "changed_gt_0_01": 0,
                "sum_abs_difference": 0.0,
                "sum_squared_difference": 0.0,
                "maximum_absolute_difference": 0.0,
                "paired_valid_reflectance_pixels": 0,
                "valid_changed_gt_0_01": 0,
                "valid_sum_abs_difference": 0.0,
                "valid_sum_squared_difference": 0.0,
            }
            for name in band_names
        }
        boundary_pixels = common_valid_pixels = 0

        # Nearest-neighbour resampling preserves the original sample values, so
        # the threshold counts below stay meaningful.
        with WarpedVRT(
            after_ds,
            crs=before_ds.crs,
            transform=before_ds.transform,
            width=before_ds.width,
            height=before_ds.height,
            resampling=Resampling.nearest,
        ) as aligned_after:
            for _, window in before_ds.block_windows(1):
                old = before_ds.read(window=window)
                new = aligned_after.read(window=window)
                inside = geometry_mask(
                    shapes,
                    out_shape=(int(window.height), int(window.width)),
                    transform=before_ds.window_transform(window),
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
                    stats["before_sentinel_like"] += int(
                        np.count_nonzero(av < SENTINEL_THRESHOLD)
                    )
                    stats["after_lt_0"] += int(np.count_nonzero(bv < 0))
                    stats["after_gt_1"] += int(np.count_nonzero(bv > 1))
                    stats["after_sentinel_like"] += int(
                        np.count_nonzero(bv < SENTINEL_THRESHOLD)
                    )
                    delta = bv.astype(np.float64) - av.astype(np.float64)
                    abs_delta = np.abs(delta)
                    stats["changed_gt_1e-6"] += int(np.count_nonzero(abs_delta > 1e-6))
                    stats["changed_gt_0_01"] += int(np.count_nonzero(abs_delta > 0.01))
                    stats["sum_abs_difference"] += float(abs_delta.sum(dtype=np.float64))
                    stats["sum_squared_difference"] += float(
                        np.square(delta).sum(dtype=np.float64)
                    )
                    if abs_delta.size:
                        stats["maximum_absolute_difference"] = max(
                            stats["maximum_absolute_difference"], float(abs_delta.max())
                        )
                    # Exclude invalid or out-of-range reflectance so the -9999
                    # placeholder cannot dominate the error metrics.
                    in_range = (av >= 0) & (av <= 1) & (bv >= 0) & (bv <= 1)
                    valid_delta = delta[in_range]
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
                if valid_count
                else None
            )

        return {
            "method": (
                "Repaired raster resampled onto the pre-repair grid with nearest "
                "neighbour; statistics use an identical boundary mask and the "
                "common alpha-valid pixels."
            ),
            "before_raster": str(before),
            "after_raster": str(after),
            "crs": str(before_ds.crs),
            "comparison_grid": {
                "width": before_ds.width,
                "height": before_ds.height,
                "pixel_size_x_m": abs(before_ds.transform.a),
                "pixel_size_y_m": abs(before_ds.transform.e),
                "transform": list(before_ds.transform),
            },
            "boundary_pixels_on_comparison_grid": boundary_pixels,
            "common_alpha_valid_boundary_pixels": common_valid_pixels,
            "bands": totals,
        }


def _make_patch_figure(
    before: Path,
    after: Path,
    output_figure: Path,
    window: Window,
    band: int,
    display_range: tuple[float, float],
) -> None:
    with rasterio.open(before) as before_ds, rasterio.open(after) as after_ds:
        bounds = before_ds.window_bounds(window)
        extent = (bounds[0], bounds[2], bounds[1], bounds[3])
        old = before_ds.read(band, window=window)
        with WarpedVRT(
            after_ds,
            crs=before_ds.crs,
            transform=before_ds.transform,
            width=before_ds.width,
            height=before_ds.height,
            resampling=Resampling.nearest,
        ) as aligned_after:
            new = aligned_after.read(band, window=window)
        crs_label = str(before_ds.crs)

    figure, axes = plt.subplots(1, 2, figsize=(11, 6.3), sharex=True, sharey=True)
    for ax, image, title in zip(
        axes, (old, new), ("Before repair", "After repair"), strict=True
    ):
        display = np.ma.masked_where(~np.isfinite(image) | (image < SENTINEL_THRESHOLD), image)
        ax.imshow(
            display,
            cmap="gray",
            vmin=display_range[0],
            vmax=display_range[1],
            extent=extent,
            interpolation="nearest",
            origin="upper",
        )
        ax.set_title(title)
        ax.set_xlabel(f"Easting (m, {crs_label})", labelpad=9)
        ax.ticklabel_format(useOffset=False, style="plain")
        ax.tick_params(axis="both", labelsize=9, pad=4)
    axes[0].set_ylabel(f"Northing (m, {crs_label})", labelpad=9)
    figure.suptitle(
        f"Band {band} reflectance: identical geographic extent and grid\n"
        f"Display range {display_range[0]}-{display_range[1]}; "
        "values above the range appear white"
    )
    figure.tight_layout(rect=(0.04, 0.14, 0.99, 0.87))
    output_figure.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_figure, dpi=200, bbox_inches="tight", pad_inches=0.16)
    plt.close(figure)
