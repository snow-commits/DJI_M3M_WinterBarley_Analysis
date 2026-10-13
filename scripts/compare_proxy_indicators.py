"""比较无生物量或产量实测值时的相对植被代理指标方案。"""

from __future__ import annotations

import argparse
import csv
from itertools import combinations
from pathlib import Path

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
import numpy as np
import rasterio
from rasterio.enums import Resampling


INDEX_RANGES = {
    "ndvi": (-1.0, 1.0),
    "ndre": (-1.0, 1.0),
    "evi2": (-2.5 / 3.4, 1.25),
}
INDEX_ORDER = ("ndvi", "ndre", "evi2")
HISTOGRAM_BINS = 100_000
SCORE_RANGE = (0.0, 100.0)
SCORE_NODATA = -9999.0
CLASS_NODATA = 0


# 单指数百分位秩作为基线；综合分数为田内百分位秩均值，不是原始指数均值。
CORE_SCENARIOS = {
    "ndvi_rank": {
        "label": "NDVI percentile-rank baseline",
        "formula": "100 * rank(NDVI)",
        "weights": {"ndvi": 1.0},
        "type": "single-index baseline",
    },
    "ndre_rank": {
        "label": "NDRE percentile-rank baseline",
        "formula": "100 * rank(NDRE)",
        "weights": {"ndre": 1.0},
        "type": "single-index baseline",
    },
    "evi2_rank": {
        "label": "EVI2 percentile-rank baseline",
        "formula": "100 * rank(EVI2)",
        "weights": {"evi2": 1.0},
        "type": "single-index baseline",
    },
    "proxy_ndvi_ndre_50_50": {
        "label": "NDVI + NDRE equal-weight proxy",
        "formula": "50 * rank(NDVI) + 50 * rank(NDRE)",
        "weights": {"ndvi": 0.5, "ndre": 0.5},
        "type": "two-index composite proxy",
    },
    "proxy_ndvi_evi2_50_50": {
        "label": "NDVI + EVI2 equal-weight proxy",
        "formula": "50 * rank(NDVI) + 50 * rank(EVI2)",
        "weights": {"ndvi": 0.5, "evi2": 0.5},
        "type": "two-index composite proxy",
    },
    "proxy_evi2_ndre_50_50": {
        "label": "EVI2 + NDRE equal-weight proxy",
        "formula": "50 * rank(EVI2) + 50 * rank(NDRE)",
        "weights": {"evi2": 0.5, "ndre": 0.5},
        "type": "two-index composite proxy",
    },
    "proxy_all_three_equal": {
        "label": "NDVI + NDRE + EVI2 equal-weight proxy",
        "formula": "100/3 * [rank(NDVI) + rank(NDRE) + rank(EVI2)]",
        "weights": {"ndvi": 1 / 3, "ndre": 1 / 3, "evi2": 1 / 3},
        "type": "three-index composite proxy",
    },
}

WEIGHT_SCENARIOS = {
    "ndvi25_ndre75": {
        "label": "NDVI 25% + NDRE 75%",
        "weights": {"ndvi": 0.25, "ndre": 0.75},
        "baseline": "proxy_ndvi_ndre_50_50",
    },
    "ndvi75_ndre25": {
        "label": "NDVI 75% + NDRE 25%",
        "weights": {"ndvi": 0.75, "ndre": 0.25},
        "baseline": "proxy_ndvi_ndre_50_50",
    },
    "evi2_25_ndre75": {
        "label": "EVI2 25% + NDRE 75%",
        "weights": {"evi2": 0.25, "ndre": 0.75},
        "baseline": "proxy_evi2_ndre_50_50",
    },
    "evi2_75_ndre25": {
        "label": "EVI2 75% + NDRE 25%",
        "weights": {"evi2": 0.75, "ndre": 0.25},
        "baseline": "proxy_evi2_ndre_50_50",
    },
}

ALL_SCENARIOS = {**CORE_SCENARIOS, **WEIGHT_SCENARIOS}
COMPOSITE_MAPS = (
    "proxy_ndvi_ndre_50_50",
    "proxy_ndvi_evi2_50_50",
    "proxy_evi2_ndre_50_50",
    "proxy_all_three_equal",
)


def _valid_mask(dataset: rasterio.io.DatasetReader, values: np.ndarray) -> np.ndarray:
    valid = np.isfinite(values)
    if dataset.nodata is not None:
        valid &= values != dataset.nodata
    return valid


def _assert_same_grid(reference: rasterio.io.DatasetReader, other: rasterio.io.DatasetReader) -> None:
    if (
        reference.width,
        reference.height,
        reference.transform,
        reference.crs,
    ) != (other.width, other.height, other.transform, other.crs):
        raise ValueError(f"Input rasters do not share the same grid: {other.name}")


def _histogram_bin_ids(values: np.ndarray, low: float, high: float) -> np.ndarray:
    ids = np.floor((values - low) * (HISTOGRAM_BINS / (high - low))).astype(np.int64)
    return np.clip(ids, 0, HISTOGRAM_BINS - 1)


def _percentile_rank(
    values: np.ndarray, histogram: np.ndarray, count: int, value_range: tuple[float, float]
) -> np.ndarray:
    """用直方图中秩估算经验百分位，返回范围为 [0, 1]。"""
    low, high = value_range
    ids = _histogram_bin_ids(values, low, high)
    cumulative = np.cumsum(histogram, dtype=np.int64)
    midrank = cumulative[ids] - 0.5 * histogram[ids]
    return np.clip(midrank / count, 0.0, 1.0)


def _score_histogram_bin_ids(values: np.ndarray) -> np.ndarray:
    return _histogram_bin_ids(values, *SCORE_RANGE)


def _score_quantile(histogram: np.ndarray, q: float, count: int) -> float:
    target = q * (count - 1)
    cumulative = np.cumsum(histogram, dtype=np.int64)
    bin_index = int(np.searchsorted(cumulative, target + 1, side="left"))
    width = (SCORE_RANGE[1] - SCORE_RANGE[0]) / HISTOGRAM_BINS
    return float(SCORE_RANGE[0] + (bin_index + 0.5) * width)


def _make_scores(ranks: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    scores: dict[str, np.ndarray] = {}
    for name, spec in ALL_SCENARIOS.items():
        score = np.zeros_like(next(iter(ranks.values())), dtype=np.float64)
        for index_name, weight in spec["weights"].items():
            score += weight * ranks[index_name]
        scores[name] = 100.0 * score
    return scores


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"No rows to write to {path}")
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class PairAccumulator:
    def __init__(self) -> None:
        self.n = 0
        self.sx = self.sy = self.sx2 = self.sy2 = self.sxy = 0.0
        self.agree = 0
        self.low_intersection = self.low_union = 0
        self.high_intersection = self.high_union = 0

    def add(
        self,
        x: np.ndarray,
        y: np.ndarray,
        class_x: np.ndarray,
        class_y: np.ndarray,
    ) -> None:
        self.n += int(x.size)
        self.sx += float(x.sum(dtype=np.float64))
        self.sy += float(y.sum(dtype=np.float64))
        self.sx2 += float(np.square(x).sum(dtype=np.float64))
        self.sy2 += float(np.square(y).sum(dtype=np.float64))
        self.sxy += float(np.multiply(x, y).sum(dtype=np.float64))
        self.agree += int(np.count_nonzero(class_x == class_y))
        low_x, low_y = class_x == 1, class_y == 1
        high_x, high_y = class_x == 5, class_y == 5
        self.low_intersection += int(np.count_nonzero(low_x & low_y))
        self.low_union += int(np.count_nonzero(low_x | low_y))
        self.high_intersection += int(np.count_nonzero(high_x & high_y))
        self.high_union += int(np.count_nonzero(high_x | high_y))

    def row(self, label_x: str, label_y: str) -> dict[str, object]:
        if self.n == 0:
            raise ValueError("Cannot summarize an empty pair comparison")
        covariance = self.sxy - self.sx * self.sy / self.n
        variance_x = max(self.sx2 - self.sx * self.sx / self.n, 0.0)
        variance_y = max(self.sy2 - self.sy * self.sy / self.n, 0.0)
        rho = covariance / np.sqrt(variance_x * variance_y) if variance_x and variance_y else float("nan")
        return {
            "candidate_a": label_x,
            "candidate_b": label_y,
            "common_pixels": self.n,
            "spearman_rho_histogram_midranks": float(rho),
            "same_quintile_percent": 100.0 * self.agree / self.n,
            "lowest_quintile_jaccard": (
                self.low_intersection / self.low_union if self.low_union else float("nan")
            ),
            "highest_quintile_jaccard": (
                self.high_intersection / self.high_union if self.high_union else float("nan")
            ),
        }


def run_experiment(index_dir: Path, boundary_path: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {name: index_dir / f"winterbarley_{name}.tif" for name in INDEX_ORDER}
    histograms = {name: np.zeros(HISTOGRAM_BINS, dtype=np.int64) for name in INDEX_ORDER}
    valid_count = 0

    # 第一步：核对共同掩膜，并统计全田指数直方图。
    with rasterio.open(paths["ndvi"]) as ndvi_ds, rasterio.open(paths["ndre"]) as ndre_ds, rasterio.open(paths["evi2"]) as evi2_ds:
        datasets = {"ndvi": ndvi_ds, "ndre": ndre_ds, "evi2": evi2_ds}
        for dataset in (ndre_ds, evi2_ds):
            _assert_same_grid(ndvi_ds, dataset)
        for _, window in ndvi_ds.block_windows(1):
            values = {name: ds.read(1, window=window) for name, ds in datasets.items()}
            masks = {name: _valid_mask(datasets[name], values[name]) for name in INDEX_ORDER}
            if not (np.array_equal(masks["ndvi"], masks["ndre"]) and np.array_equal(masks["ndvi"], masks["evi2"])):
                raise ValueError("NDVI, NDRE, and EVI2 masks differ; align to an explicit common mask first.")
            mask = masks["ndvi"]
            n = int(np.count_nonzero(mask))
            valid_count += n
            for name in INDEX_ORDER:
                low, high = INDEX_RANGES[name]
                block_hist, _ = np.histogram(values[name][mask], bins=HISTOGRAM_BINS, range=(low, high))
                histograms[name] += block_hist

        if valid_count == 0:
            raise ValueError("No common valid pixels found in the three index rasters.")

        # 初始化候选栅格；输出为 0–100 相对分数，不代表反射率、生物量或产量。
        output_paths = {name: output_dir / f"{name}.tif" for name in COMPOSITE_MAPS}
        profile = ndvi_ds.profile.copy()
        profile.update(dtype="float32", nodata=SCORE_NODATA, compress="deflate", predictor=3)
        destinations = {}
        for name, dst_path in output_paths.items():
            dst = rasterio.open(dst_path, "w", **profile)
            dst.set_band_description(1, f"{CORE_SCENARIOS[name]['label']} score (0-100)")
            dst.update_tags(
                1,
                PROXY_TYPE=CORE_SCENARIOS[name]["type"],
                FORMULA=CORE_SCENARIOS[name]["formula"],
                RANK_METHOD=f"{HISTOGRAM_BINS}-bin histogram midpoint percentile estimate",
                SCORE_SCALE="0-100 within-field relative ranking; not biomass/yield units",
                VALID_MASK="Common valid mask of the input NDVI, NDRE, and EVI2 rasters",
            )
            destinations[name] = dst

        scenario_histograms = {
            name: np.zeros(HISTOGRAM_BINS, dtype=np.int64) for name in ALL_SCENARIOS
        }
        sums = {name: 0.0 for name in ALL_SCENARIOS}
        sums_sq = {name: 0.0 for name in ALL_SCENARIOS}
        minima = {name: float("inf") for name in ALL_SCENARIOS}
        maxima = {name: float("-inf") for name in ALL_SCENARIOS}

        try:
            # 第二步：将指数转换为百分位，生成候选分数并写出四种综合栅格。
            for _, window in ndvi_ds.block_windows(1):
                values = {name: ds.read(1, window=window) for name, ds in datasets.items()}
                mask = _valid_mask(ndvi_ds, values["ndvi"])
                arrays = {name: values[name][mask].astype(np.float64, copy=False) for name in INDEX_ORDER}
                ranks = {
                    name: _percentile_rank(arrays[name], histograms[name], valid_count, INDEX_RANGES[name])
                    for name in INDEX_ORDER
                }
                scores = _make_scores(ranks)
                for name, score in scores.items():
                    if score.size:
                        scenario_histograms[name] += np.histogram(
                            score, bins=HISTOGRAM_BINS, range=SCORE_RANGE
                        )[0]
                        sums[name] += float(score.sum(dtype=np.float64))
                        sums_sq[name] += float(np.square(score).sum(dtype=np.float64))
                        minima[name] = min(minima[name], float(score.min()))
                        maxima[name] = max(maxima[name], float(score.max()))
                for name in COMPOSITE_MAPS:
                    out = np.full(mask.shape, SCORE_NODATA, dtype=np.float32)
                    if np.any(mask):
                        out[mask] = scores[name].astype(np.float32, copy=False)
                    destinations[name].write(out, 1, window=window)
        finally:
            for dst in destinations.values():
                dst.close()

    # 汇总统计量及各方案的五分位分级阈值。
    pixel_area_m2 = abs(ndvi_ds.transform.a * ndvi_ds.transform.e - ndvi_ds.transform.b * ndvi_ds.transform.d)
    summary_rows: list[dict[str, object]] = []
    thresholds: dict[str, list[float]] = {}
    for name, spec in ALL_SCENARIOS.items():
        hist = scenario_histograms[name]
        thresholds[name] = [_score_quantile(hist, q, valid_count) for q in (0.2, 0.4, 0.6, 0.8)]
        mean = sums[name] / valid_count
        variance = max(sums_sq[name] / valid_count - mean * mean, 0.0)
        q05, q20, q40, q50, q60, q80, q95 = [
            _score_quantile(hist, q, valid_count) for q in (0.05, 0.2, 0.4, 0.5, 0.6, 0.8, 0.95)
        ]
        summary_rows.append(
            {
                "candidate": name,
                "label": spec["label"],
                "candidate_type": spec["type"] if "type" in spec else "weight sensitivity scenario",
                "formula": spec.get("formula", "weighted mean of within-field percentile ranks"),
                "weights": "; ".join(f"{k.upper()}={v:.2f}" for k, v in spec["weights"].items()),
                "valid_pixels": valid_count,
                "mean_score_0_100": mean,
                "stddev_score": float(np.sqrt(variance)),
                "p05_score": q05,
                "q20_score": q20,
                "q40_score": q40,
                "median_score": q50,
                "q60_score": q60,
                "q80_score": q80,
                "p95_score": q95,
                "p95_minus_p05": q95 - q05,
                "minimum_score": minima[name],
                "maximum_score": maxima[name],
            }
        )
    _write_csv(output_dir / "proxy_candidate_summary.csv", summary_rows)

    # 第三步：在共同像元上比较地图；代理分数再转百分位以估算 Spearman 相关。
    # 等级一致率和 Jaccard 指数比较各方案自身的五分位等级，重点对比四种综合实验。
    core_pairs = list(combinations(COMPOSITE_MAPS, 2))
    pair_accumulators = {pair: PairAccumulator() for pair in core_pairs}
    sensitivity_pairs = {
        (spec["baseline"], name): PairAccumulator() for name, spec in WEIGHT_SCENARIOS.items()
    }
    class_counts = {name: np.zeros(5, dtype=np.int64) for name in COMPOSITE_MAPS}
    class_destinations = {}
    with rasterio.open(paths["ndvi"]) as ndvi_ds, rasterio.open(paths["ndre"]) as ndre_ds, rasterio.open(paths["evi2"]) as evi2_ds:
        datasets = {"ndvi": ndvi_ds, "ndre": ndre_ds, "evi2": evi2_ds}
        profile = ndvi_ds.profile.copy()
        profile.update(dtype="uint8", count=1, nodata=CLASS_NODATA, compress="deflate", predictor=2)
        for name in COMPOSITE_MAPS:
            class_path = output_dir / f"{name}_quintile.tif"
            dst = rasterio.open(class_path, "w", **profile)
            dst.set_band_description(1, f"{CORE_SCENARIOS[name]['label']} quintile class")
            dst.update_tags(
                1,
                CLASSIFICATION="Five equal-frequency classes within this field; relative zones only, not agronomic thresholds",
                CLASS_CODES="1=lowest; 2=low; 3=middle; 4=high; 5=highest; 0=NoData",
                QUANTILE_THRESHOLDS=";".join(f"{v:.6f}" for v in thresholds[name]),
            )
            class_destinations[name] = dst

        try:
            for _, window in ndvi_ds.block_windows(1):
                values = {name: ds.read(1, window=window) for name, ds in datasets.items()}
                mask = _valid_mask(ndvi_ds, values["ndvi"])
                arrays = {name: values[name][mask].astype(np.float64, copy=False) for name in INDEX_ORDER}
                ranks = {
                    name: _percentile_rank(arrays[name], histograms[name], valid_count, INDEX_RANGES[name])
                    for name in INDEX_ORDER
                }
                scores = _make_scores(ranks)
                ranks_of_scores = {
                    name: _percentile_rank(scores[name], scenario_histograms[name], valid_count, SCORE_RANGE)
                    for name in ALL_SCENARIOS
                }
                classes = {
                    name: np.digitize(scores[name], thresholds[name], right=False).astype(np.uint8) + 1
                    for name in ALL_SCENARIOS
                }
                for name in COMPOSITE_MAPS:
                    if classes[name].size:
                        class_counts[name] += np.bincount(classes[name] - 1, minlength=5)[:5]
                for (name_a, name_b), accumulator in pair_accumulators.items():
                    accumulator.add(ranks_of_scores[name_a], ranks_of_scores[name_b], classes[name_a], classes[name_b])
                for (baseline, sensitivity), accumulator in sensitivity_pairs.items():
                    accumulator.add(
                        ranks_of_scores[baseline],
                        ranks_of_scores[sensitivity],
                        classes[baseline],
                        classes[sensitivity],
                    )
                for name, dst in class_destinations.items():
                    out = np.zeros(mask.shape, dtype=np.uint8)
                    if np.any(mask):
                        out[mask] = classes[name]
                    dst.write(out, 1, window=window)
        finally:
            for dst in class_destinations.values():
                dst.close()

    pair_rows = [acc.row(a, b) for (a, b), acc in pair_accumulators.items()]
    _write_csv(output_dir / "proxy_candidate_pairwise_comparison.csv", pair_rows)
    weight_rows = []
    for (baseline, sensitivity), acc in sensitivity_pairs.items():
        row = acc.row(CORE_SCENARIOS[baseline]["label"], WEIGHT_SCENARIOS[sensitivity]["label"])
        row["weight_scenario"] = sensitivity
        row["equal_weight_baseline"] = baseline
        weight_rows.append(row)
    _write_csv(output_dir / "proxy_weight_sensitivity.csv", weight_rows)

    class_rows: list[dict[str, object]] = []
    for name in COMPOSITE_MAPS:
        for code, count in enumerate(class_counts[name], start=1):
            class_rows.append(
                {
                    "candidate": name,
                    "label": CORE_SCENARIOS[name]["label"],
                    "class_code": code,
                    "relative_zone": ("Lowest", "Low", "Middle", "High", "Highest")[code - 1],
                    "pixel_count": int(count),
                    "area_ha": count * pixel_area_m2 / 10_000.0,
                    "area_percent": 100.0 * count / valid_count,
                }
            )
    _write_csv(output_dir / "proxy_quintile_area.csv", class_rows)

    _create_maps(output_paths, boundary_path, output_dir, ndvi_ds.crs)
    notes = (
        "Proxy indicator experiment\n"
        "\n"
        "This is an exploratory within-field relative ranking experiment. It does not estimate measured biomass or yield.\n"
        "The four main experiments are equal-weight rank composites: NDVI+NDRE, NDVI+EVI2, EVI2+NDRE, and all three indices.\n"
        "The three two-index experiments cover every unique pair from NDVI, NDRE, and EVI2.\n"
        "The main pairwise comparison table compares only these four composites (six pairwise contrasts).\n"
        "No ground-truth biomass/yield observations are present, so predictive R2, RMSE, and MAE are not applicable.\n"
        f"Input ranks use {HISTOGRAM_BINS:,}-bin histogram midpoint estimates on the common valid mask ({valid_count:,} pixels).\n"
        "Composite weights are transparent heuristic scenarios, not fitted parameters.\n"
        "The quintiles are relative to this field/date and are not agronomic thresholds.\n"
        "Pairwise Spearman rho is estimated from histogram midranks; class agreement and Jaccard compare relative quintile zones.\n"
        "For flight-striping, inspect the score maps against the original imagery/flight direction; these statistics do not establish artifact-free accuracy.\n"
    )
    (output_dir / "README.txt").write_text(notes, encoding="utf-8")
    print(f"Wrote proxy comparison outputs to {output_dir}")
    print(f"Common valid pixels: {valid_count:,}; pixel area: {pixel_area_m2:.6f} m2")
    for row in pair_rows:
        print(
            f"{row['candidate_a']} vs {row['candidate_b']}: "
            f"rho={row['spearman_rho_histogram_midranks']:.4f}, "
            f"same quintile={row['same_quintile_percent']:.2f}%"
        )


def _create_maps(
    score_paths: dict[str, Path], boundary_path: Path, output_dir: Path, raster_crs: object
) -> None:
    boundary = gpd.read_file(boundary_path).to_crs(raster_crs)
    colors = ["#a50026", "#f46d43", "#fee08b", "#a6d96a", "#1a9850"]
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(np.arange(0.5, 6.5, 1), cmap.N)
    titles = {
        "proxy_ndvi_ndre_50_50": "NDVI + NDRE (50:50)",
        "proxy_ndvi_evi2_50_50": "NDVI + EVI2 (50:50)",
        "proxy_evi2_ndre_50_50": "EVI2 + NDRE (50:50)",
        "proxy_all_three_equal": "NDVI + NDRE + EVI2 (equal)",
    }

    fig, axes = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)
    for ax, name in zip(axes.flat, COMPOSITE_MAPS):
        with rasterio.open(score_paths[name]) as src:
            factor = 10
            out_shape = (max(1, src.height // factor), max(1, src.width // factor))
            image = src.read(1, out_shape=out_shape, masked=True, resampling=Resampling.average)
            extent = (src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top)
        im = ax.imshow(image, extent=extent, origin="upper", cmap="viridis", vmin=0, vmax=100)
        boundary.boundary.plot(ax=ax, color="#202020", linewidth=0.8)
        ax.set_title(titles[name], fontsize=11)
        ax.set_xlabel("Easting (m)", fontsize=9)
        ax.set_ylabel("Northing (m)", fontsize=9)
        ax.tick_params(labelsize=8)
        ax.set_aspect("equal")
        cbar = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
        cbar.set_label("Relative proxy score (0-100)", fontsize=8)
        cbar.ax.tick_params(labelsize=8)
    fig.suptitle("Relative Vegetation Proxy Scores\nNot calibrated biomass or yield", fontsize=14)
    fig.savefig(output_dir / "proxy_composite_continuous_maps.png", dpi=240, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)
    for ax, name in zip(axes.flat, COMPOSITE_MAPS):
        class_path = output_dir / f"{name}_quintile.tif"
        with rasterio.open(class_path) as src:
            factor = 10
            out_shape = (max(1, src.height // factor), max(1, src.width // factor))
            image = src.read(1, out_shape=out_shape, masked=True, resampling=Resampling.nearest)
            extent = (src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top)
        ax.imshow(image, extent=extent, origin="upper", cmap=cmap, norm=norm, interpolation="nearest")
        boundary.boundary.plot(ax=ax, color="#202020", linewidth=0.8)
        ax.set_title(titles[name], fontsize=11)
        ax.set_xlabel("Easting (m)", fontsize=9)
        ax.set_ylabel("Northing (m)", fontsize=9)
        ax.tick_params(labelsize=8)
        ax.set_aspect("equal")
        mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        cbar = fig.colorbar(mappable, ax=ax, fraction=0.045, pad=0.02, ticks=[1, 2, 3, 4, 5])
        cbar.ax.set_yticklabels(["Lowest", "Low", "Middle", "High", "Highest"])
        cbar.set_label("Within-field relative rank", fontsize=8)
        cbar.ax.tick_params(labelsize=8)
    fig.suptitle("Proxy Composite Relative Quintiles\nEqual-frequency classes; not agronomic thresholds", fontsize=14)
    fig.savefig(output_dir / "proxy_composite_quintile_maps.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index-dir", type=Path, required=True)
    parser.add_argument("--boundary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--figures-only",
        action="store_true",
        help="Regenerate the comparison figures from existing score and quintile rasters.",
    )
    args = parser.parse_args()
    if args.figures_only:
        with rasterio.open(args.index_dir / "winterbarley_ndvi.tif") as reference:
            crs = reference.crs
        score_paths = {name: args.output_dir / f"{name}.tif" for name in COMPOSITE_MAPS}
        _create_maps(score_paths, args.boundary, args.output_dir, crs)
        return
    run_experiment(args.index_dir, args.boundary, args.output_dir)


if __name__ == "__main__":
    main()
