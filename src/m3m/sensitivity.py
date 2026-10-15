"""Sensitivity of within-field relative scores to the choice of index weights.

.. warning::

   This module measures **weight-scheme sensitivity**, not predictive accuracy.
   It takes the *same* information (the percentile rank of each index within the
   field) and re-weights it, then reports how similar the resulting maps are.
   Two candidate scores agreeing does not mean either is closer to biomass or
   yield: with no ground-truth observations in the comparison there is nothing
   for that agreement to be measured against.

   Read the outputs as "how robust is my zoning to the weights I picked", never
   as "this proxy predicts biomass". The ranking is relative to one field on one
   date and is not an agronomic threshold.
"""

from __future__ import annotations

import csv
from itertools import combinations
from pathlib import Path
from typing import Any

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import rasterio  # noqa: E402
from matplotlib.colors import BoundaryNorm, ListedColormap  # noqa: E402
from rasterio.enums import Resampling  # noqa: E402

from m3m.constants import HISTOGRAM_BINS, INDEX_ORDER, INDEX_RANGES, NODATA  # noqa: E402
from m3m.io import assert_same_grid, valid_mask  # noqa: E402
from m3m.zones import require_index_rasters  # noqa: E402

SCORE_RANGE = (0.0, 100.0)
SCORE_NODATA = NODATA
CLASS_NODATA = 0

#: Single-index percentile-rank baselines plus the equal-weight composites that
#: form the main comparison table.
CORE_SCENARIOS: dict[str, dict[str, Any]] = {
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

#: Off-equal weight variants used to test how much the 50:50 choice matters.
WEIGHT_SCENARIOS: dict[str, dict[str, Any]] = {
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

ALL_SCENARIOS: dict[str, dict[str, Any]] = {**CORE_SCENARIOS, **WEIGHT_SCENARIOS}

#: The four composites written out as rasters and compared pairwise.
COMPOSITE_MAPS: tuple[str, ...] = (
    "proxy_ndvi_ndre_50_50",
    "proxy_ndvi_evi2_50_50",
    "proxy_evi2_ndre_50_50",
    "proxy_all_three_equal",
)

__all__ = [
    "ALL_SCENARIOS",
    "COMPOSITE_MAPS",
    "CORE_SCENARIOS",
    "WEIGHT_SCENARIOS",
    "PairAccumulator",
    "create_maps",
    "run_experiment",
]


def _histogram_bin_ids(values: np.ndarray, low: float, high: float) -> np.ndarray:
    ids = np.floor((values - low) * (HISTOGRAM_BINS / (high - low))).astype(np.int64)
    return np.clip(ids, 0, HISTOGRAM_BINS - 1)


def _percentile_rank(
    values: np.ndarray,
    histogram: np.ndarray,
    count: int,
    value_range: tuple[float, float],
) -> np.ndarray:
    """Estimate the empirical percentile from a histogram; returns values in [0, 1]."""
    low, high = value_range
    ids = _histogram_bin_ids(values, low, high)
    cumulative = np.cumsum(histogram, dtype=np.int64)
    midrank = cumulative[ids] - 0.5 * histogram[ids]
    return np.clip(midrank / count, 0.0, 1.0)


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


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"No rows to write to {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class PairAccumulator:
    """Streaming pairwise statistics between two candidate score maps."""

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

    def row(self, label_x: str, label_y: str) -> dict[str, Any]:
        if self.n == 0:
            raise ValueError("Cannot summarize an empty pair comparison")
        covariance = self.sxy - self.sx * self.sy / self.n
        variance_x = max(self.sx2 - self.sx * self.sx / self.n, 0.0)
        variance_y = max(self.sy2 - self.sy * self.sy / self.n, 0.0)
        rho = (
            covariance / np.sqrt(variance_x * variance_y)
            if variance_x and variance_y
            else float("nan")
        )
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


def _open_all(paths: dict[str, Path]) -> list[rasterio.io.DatasetReader]:
    return [rasterio.open(paths[name]) for name in INDEX_ORDER]


def _accumulate_histograms(
    datasets: dict[str, rasterio.io.DatasetReader],
) -> tuple[dict[str, np.ndarray], int]:
    """Verify the shared mask and accumulate whole-field index histograms."""
    reference = datasets[INDEX_ORDER[0]]
    histograms = {name: np.zeros(HISTOGRAM_BINS, dtype=np.int64) for name in INDEX_ORDER}
    valid_count = 0

    for _, window in reference.block_windows(1):
        values = {name: ds.read(1, window=window) for name, ds in datasets.items()}
        masks = {name: valid_mask(datasets[name], values[name]) for name in INDEX_ORDER}
        first = masks[INDEX_ORDER[0]]
        for name in INDEX_ORDER[1:]:
            if not np.array_equal(first, masks[name]):
                raise ValueError(
                    f"{name.upper()} and {INDEX_ORDER[0].upper()} valid masks differ; "
                    "align the index rasters to an explicit common mask first."
                )
        n = int(np.count_nonzero(first))
        valid_count += n
        for name in INDEX_ORDER:
            low, high = INDEX_RANGES[name]
            block_histogram, _ = np.histogram(
                values[name][first], bins=HISTOGRAM_BINS, range=(low, high)
            )
            histograms[name] += block_histogram

    if valid_count == 0:
        raise ValueError("No common valid pixels found in the index rasters.")
    return histograms, valid_count


def _ranks_for_window(
    values: dict[str, np.ndarray],
    mask: np.ndarray,
    histograms: dict[str, np.ndarray],
    valid_count: int,
) -> dict[str, np.ndarray]:
    arrays = {
        name: values[name][mask].astype(np.float64, copy=False) for name in INDEX_ORDER
    }
    return {
        name: _percentile_rank(arrays[name], histograms[name], valid_count, INDEX_RANGES[name])
        for name in INDEX_ORDER
    }


def run_experiment(
    index_dir: Path | str,
    boundary_path: Path | str,
    output_dir: Path | str,
    *,
    prefix: str = "field",
    make_figures: bool = True,
) -> dict[str, Any]:
    """Compute every weight scenario and write the rasters, CSVs and figures."""
    index_dir = Path(index_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = require_index_rasters(index_dir, prefix)

    # ---- step 1: shared mask, whole-field histograms, output rasters ------- #
    datasets = _open_all(paths)
    try:
        reference = datasets[0]
        for dataset in datasets[1:]:
            assert_same_grid(reference, dataset)
        dataset_map = dict(zip(INDEX_ORDER, datasets, strict=True))

        histograms, valid_count = _accumulate_histograms(dataset_map)
        crs = reference.crs
        pixel_area_m2 = abs(
            reference.transform.a * reference.transform.e
            - reference.transform.b * reference.transform.d
        )

        output_paths = {name: output_dir / f"{name}.tif" for name in COMPOSITE_MAPS}
        profile = reference.profile.copy()
        profile.update(dtype="float32", nodata=SCORE_NODATA, compress="deflate", predictor=3)
        destinations: dict[str, rasterio.io.DatasetWriter] = {}
        for name, dst_path in output_paths.items():
            dst = rasterio.open(dst_path, "w", **profile)
            dst.set_band_description(1, f"{CORE_SCENARIOS[name]['label']} score (0-100)")
            dst.update_tags(
                1,
                PROXY_TYPE=CORE_SCENARIOS[name]["type"],
                FORMULA=CORE_SCENARIOS[name]["formula"],
                RANK_METHOD=f"{HISTOGRAM_BINS}-bin histogram midpoint percentile estimate",
                SCORE_SCALE=(
                    "0-100 within-field relative ranking; not biomass or yield units"
                ),
                VALID_MASK="Common valid mask of the input NDVI, NDRE and EVI2 rasters",
                INTERPRETATION=(
                    "Weight-scheme sensitivity experiment; no ground truth is used and "
                    "no predictive accuracy is claimed."
                ),
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
            # ---- step 2: ranks -> scores -> score rasters ------------------ #
            for _, window in reference.block_windows(1):
                values = {name: ds.read(1, window=window) for name, ds in dataset_map.items()}
                mask = valid_mask(reference, values[INDEX_ORDER[0]])
                ranks = _ranks_for_window(values, mask, histograms, valid_count)
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
    finally:
        for dataset in datasets:
            dataset.close()

    # ---- summary statistics and per-scenario quintile thresholds ---------- #
    summary_rows: list[dict[str, Any]] = []
    thresholds: dict[str, list[float]] = {}
    for name, spec in ALL_SCENARIOS.items():
        histogram = scenario_histograms[name]
        thresholds[name] = [
            _score_quantile(histogram, q, valid_count) for q in (0.2, 0.4, 0.6, 0.8)
        ]
        mean = sums[name] / valid_count
        variance = max(sums_sq[name] / valid_count - mean * mean, 0.0)
        q05, q20, q40, q50, q60, q80, q95 = (
            _score_quantile(histogram, q, valid_count)
            for q in (0.05, 0.2, 0.4, 0.5, 0.6, 0.8, 0.95)
        )
        summary_rows.append(
            {
                "candidate": name,
                "label": spec["label"],
                "candidate_type": spec.get("type", "weight sensitivity scenario"),
                "formula": spec.get(
                    "formula", "weighted mean of within-field percentile ranks"
                ),
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

    # ---- step 3: pairwise comparison on the common pixels ----------------- #
    core_pairs = list(combinations(COMPOSITE_MAPS, 2))
    pair_accumulators = {pair: PairAccumulator() for pair in core_pairs}
    sensitivity_pairs = {
        (spec["baseline"], name): PairAccumulator() for name, spec in WEIGHT_SCENARIOS.items()
    }
    class_counts = {name: np.zeros(5, dtype=np.int64) for name in COMPOSITE_MAPS}

    datasets = _open_all(paths)
    class_destinations: dict[str, rasterio.io.DatasetWriter] = {}
    try:
        reference = datasets[0]
        dataset_map = dict(zip(INDEX_ORDER, datasets, strict=True))
        profile = reference.profile.copy()
        profile.update(dtype="uint8", count=1, nodata=CLASS_NODATA, compress="deflate", predictor=2)
        for name in COMPOSITE_MAPS:
            class_path = output_dir / f"{name}_quintile.tif"
            dst = rasterio.open(class_path, "w", **profile)
            dst.set_band_description(1, f"{CORE_SCENARIOS[name]['label']} quintile class")
            dst.update_tags(
                1,
                CLASSIFICATION=(
                    "Five equal-frequency classes within this field; relative zones "
                    "only, not agronomic thresholds"
                ),
                CLASS_CODES="1=lowest; 2=low; 3=middle; 4=high; 5=highest; 0=NoData",
                QUANTILE_THRESHOLDS=";".join(f"{v:.6f}" for v in thresholds[name]),
            )
            class_destinations[name] = dst

        try:
            for _, window in reference.block_windows(1):
                values = {name: ds.read(1, window=window) for name, ds in dataset_map.items()}
                mask = valid_mask(reference, values[INDEX_ORDER[0]])
                ranks = _ranks_for_window(values, mask, histograms, valid_count)
                scores = _make_scores(ranks)
                ranks_of_scores = {
                    name: _percentile_rank(
                        scores[name], scenario_histograms[name], valid_count, SCORE_RANGE
                    )
                    for name in ALL_SCENARIOS
                }
                classes = {
                    name: np.digitize(scores[name], thresholds[name], right=False).astype(
                        np.uint8
                    )
                    + 1
                    for name in ALL_SCENARIOS
                }
                for name in COMPOSITE_MAPS:
                    if classes[name].size:
                        class_counts[name] += np.bincount(classes[name] - 1, minlength=5)[:5]
                for (name_a, name_b), accumulator in pair_accumulators.items():
                    accumulator.add(
                        ranks_of_scores[name_a],
                        ranks_of_scores[name_b],
                        classes[name_a],
                        classes[name_b],
                    )
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
    finally:
        for dataset in datasets:
            dataset.close()

    pair_rows = [accumulator.row(a, b) for (a, b), accumulator in pair_accumulators.items()]
    _write_csv(output_dir / "proxy_candidate_pairwise_comparison.csv", pair_rows)

    weight_rows = []
    for (baseline, sensitivity), accumulator in sensitivity_pairs.items():
        row = accumulator.row(
            CORE_SCENARIOS[baseline]["label"], WEIGHT_SCENARIOS[sensitivity]["label"]
        )
        row["weight_scenario"] = sensitivity
        row["equal_weight_baseline"] = baseline
        weight_rows.append(row)
    _write_csv(output_dir / "proxy_weight_sensitivity.csv", weight_rows)

    class_rows: list[dict[str, Any]] = []
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

    if make_figures:
        create_maps(output_paths, boundary_path, output_dir, crs)

    notes = (
        "Weight-sensitivity experiment for within-field relative scores\n"
        "\n"
        "WHAT THIS IS: an exploratory check of how much the relative zoning changes\n"
        "when the weights given to each vegetation index are varied. All candidates\n"
        "are built from the same index ranks, so the comparisons below are\n"
        "self-referential by construction.\n"
        "\n"
        "WHAT THIS IS NOT: a prediction of measured biomass or yield. No ground-truth\n"
        "observations are present in this dataset, so predictive R2, RMSE and MAE are\n"
        "not applicable and are deliberately not reported.\n"
        "\n"
        "The four main candidates are equal-weight rank composites: NDVI+NDRE,\n"
        "NDVI+EVI2, EVI2+NDRE, and all three indices. The four weight-sensitivity\n"
        "scenarios re-weight two-index composites at 25:75 and 75:25.\n"
        f"Ranks use {HISTOGRAM_BINS:,}-bin histogram midpoint estimates on the common\n"
        f"valid mask ({valid_count:,} pixels).\n"
        "Composite weights are transparent heuristic choices, not fitted parameters.\n"
        "The quintiles are relative to this field and date and are not agronomic\n"
        "thresholds.\n"
        "Pairwise Spearman rho is estimated from histogram midranks; class agreement\n"
        "and Jaccard indices compare the relative quintile zones.\n"
        "For flight striping, inspect the score maps against the original imagery and\n"
        "flight direction: these statistics do not establish artifact-free accuracy.\n"
    )
    (output_dir / "README.txt").write_text(notes, encoding="utf-8")

    return {
        "valid_pixels": valid_count,
        "pixel_area_m2": pixel_area_m2,
        "summary_rows": summary_rows,
        "pair_rows": pair_rows,
        "weight_rows": weight_rows,
        "class_rows": class_rows,
        "output_dir": output_dir,
    }


def create_maps(
    score_paths: dict[str, Path],
    boundary_path: Path | str,
    output_dir: Path | str,
    raster_crs: Any,
) -> None:
    """Write the continuous-score and relative-quintile comparison figures."""
    output_dir = Path(output_dir)
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

    figure, axes = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)
    for ax, name in zip(axes.flat, COMPOSITE_MAPS, strict=True):
        with rasterio.open(score_paths[name]) as src:
            factor = 10
            out_shape = (max(1, src.height // factor), max(1, src.width // factor))
            image = src.read(1, out_shape=out_shape, masked=True, resampling=Resampling.average)
            extent = (src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top)
        plotted = ax.imshow(
            image, extent=extent, origin="upper", cmap="viridis", vmin=0, vmax=100
        )
        boundary.boundary.plot(ax=ax, color="#202020", linewidth=0.8)
        ax.set_title(titles[name], fontsize=11)
        ax.set_xlabel("Easting (m)", fontsize=9)
        ax.set_ylabel("Northing (m)", fontsize=9)
        ax.tick_params(labelsize=8)
        ax.set_aspect("equal")
        colorbar = figure.colorbar(plotted, ax=ax, fraction=0.045, pad=0.02)
        colorbar.set_label("Relative score (0-100)", fontsize=8)
        colorbar.ax.tick_params(labelsize=8)
    figure.suptitle(
        "Relative Vegetation Scores\nNot calibrated biomass or yield", fontsize=14
    )
    figure.savefig(
        output_dir / "proxy_composite_continuous_maps.png", dpi=240, bbox_inches="tight"
    )
    plt.close(figure)

    figure, axes = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)
    for ax, name in zip(axes.flat, COMPOSITE_MAPS, strict=True):
        class_path = output_dir / f"{name}_quintile.tif"
        with rasterio.open(class_path) as src:
            factor = 10
            out_shape = (max(1, src.height // factor), max(1, src.width // factor))
            image = src.read(1, out_shape=out_shape, masked=True, resampling=Resampling.nearest)
            extent = (src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top)
        ax.imshow(
            image, extent=extent, origin="upper", cmap=cmap, norm=norm, interpolation="nearest"
        )
        boundary.boundary.plot(ax=ax, color="#202020", linewidth=0.8)
        ax.set_title(titles[name], fontsize=11)
        ax.set_xlabel("Easting (m)", fontsize=9)
        ax.set_ylabel("Northing (m)", fontsize=9)
        ax.tick_params(labelsize=8)
        ax.set_aspect("equal")
        mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        colorbar = figure.colorbar(mappable, ax=ax, fraction=0.045, pad=0.02, ticks=[1, 2, 3, 4, 5])
        colorbar.ax.set_yticklabels(["Lowest", "Low", "Middle", "High", "Highest"])
        colorbar.set_label("Within-field relative rank", fontsize=8)
        colorbar.ax.tick_params(labelsize=8)
    figure.suptitle(
        "Relative Quintiles per Weight Scenario\n"
        "Equal-frequency classes; not agronomic thresholds",
        fontsize=14,
    )
    figure.savefig(
        output_dir / "proxy_composite_quintile_maps.png", dpi=240, bbox_inches="tight"
    )
    plt.close(figure)
