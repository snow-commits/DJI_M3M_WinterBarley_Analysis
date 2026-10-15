"""Variable-rate prescription maps from within-field relative zones.

Turns the five relative vigour classes produced by :mod:`m3m.zones` into an
application-rate raster plus the area, product quantity and cost table a field
operation actually needs.

What decides what
-----------------
The **rate schedule is an operator input, not a model output.** Nothing in this
toolkit derives it: there is no ground truth in the pipeline, so the tool cannot
know how much product a zone needs. The schedule encodes a decision the operator
has already made (from a soil test, standard practice, or a trial result), and
this module's job is to apply it consistently and report the consequences.

The optional ``target_mean_rate`` closes the loop in the other direction: give
per-zone *relative* weights plus the field-average rate you intend to apply, and
the schedule is rescaled so the area-weighted mean hits that target. The operator
still owns the total; the tool owns the spatial distribution.

Read the equal-frequency caveat
-------------------------------
The input zones are equal-frequency quantiles, so each class covers roughly a
fifth of the field *by construction*, whatever the crop is actually doing. A
prescription built on them can redistribute product **within** a field decision
that has already been made; it can never conclude that a field needs less
overall, because one fifth of it is always in the top class. Use
``target_mean_rate`` (or a fixed absolute schedule) to keep the total under
operator control.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
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

from m3m.constants import RELATIVE_CLASS_NAMES  # noqa: E402

SUMMARY_FIELDS: tuple[str, ...] = (
    "class_code",
    "relative_class",
    "rate",
    "unit",
    "pixel_count",
    "area_ha",
    "area_percent",
    "product_quantity",
    "quantity_unit",
    "product_cost",
    "application_cost",
    "total_cost",
)

NODATA_RATE = -9999.0

ZONE_COLORS = ("#a50026", "#f46d43", "#fee08b", "#a6d96a", "#1a9850")

#: Suffixes that turn a per-hectare rate into an absolute quantity.
_PER_HECTARE_SUFFIXES = ("/ha", " per ha", "/hectare", " ha-1", " ha^-1")

__all__ = [
    "NODATA_RATE",
    "SUMMARY_FIELDS",
    "Pricing",
    "PrescriptionError",
    "PrescriptionResult",
    "RateSchedule",
    "build",
    "create_figure",
    "quantity_unit",
    "validate_schedule",
]


def quantity_unit(unit: str) -> str:
    """Derive the unit of ``rate * area`` from the rate's own unit.

    ``kg N/ha`` becomes ``kg N`` because multiplying by hectares cancels the
    divisor. Anything that is not a per-hectare rate keeps an explicit ``*ha``
    so the quantity is never silently mislabelled.
    """
    text = unit.strip()
    lowered = text.lower()
    if lowered in {"relative", "unitless", ""}:
        return "relative*ha"
    for suffix in _PER_HECTARE_SUFFIXES:
        if lowered.endswith(suffix):
            base = text[: -len(suffix)].strip()
            return base or "unit"
    return f"{text}*ha"


class PrescriptionError(ValueError):
    """Raised when a prescription cannot be built from the given inputs."""


@dataclass
class RateSchedule:
    """Application rate per relative vigour class."""

    rates: dict[int, float]
    unit: str = "relative"

    def rate_for(self, class_code: int) -> float:
        try:
            return float(self.rates[class_code])
        except KeyError as exc:
            raise PrescriptionError(
                f"No rate configured for relative class {class_code}. "
                f"Configured classes: {', '.join(str(c) for c in sorted(self.rates))}."
            ) from exc

    def rescaled(self, factor: float) -> RateSchedule:
        return RateSchedule(
            rates={code: value * factor for code, value in self.rates.items()},
            unit=self.unit,
        )

    @property
    def is_relative(self) -> bool:
        return self.unit.strip().lower() in {"relative", "unitless", ""}


@dataclass
class Pricing:
    """Optional cost model. All zeros means 'do not report money'."""

    product_price_per_unit: float = 0.0
    currency: str = "CNY"
    application_cost_per_ha: float = 0.0

    @property
    def enabled(self) -> bool:
        return self.product_price_per_unit > 0.0 or self.application_cost_per_ha > 0.0


@dataclass
class PrescriptionResult:
    """Everything ``m3m prescribe`` produced, in one object."""

    rate_raster: Path
    summary_csv: Path
    totals_json: Path
    figure: Path | None
    rows: list[dict[str, Any]]
    totals: dict[str, Any]
    warnings: list[str] = field(default_factory=list)

    def format_report(self) -> str:
        """Render the operator-facing summary."""
        unit = self.totals["unit"]
        quantity_unit_text = self.totals["quantity_unit"]
        lines = [
            f"Relative zones: {self.totals['zone_count']}",
            f"Treated area  : {self.totals['area_ha']:.3f} ha "
            f"({self.totals['valid_pixels']:,} pixels at "
            f"{self.totals['pixel_area_m2']:.6f} m2)",
            f"Mean rate     : {self.totals['mean_rate']:.4f} {unit}"
            + (
                f"  (target {self.totals['target_mean_rate']:.4f})"
                if self.totals.get("target_mean_rate")
                else ""
            ),
            f"Total product : {self.totals['product_quantity']:.4f} {quantity_unit_text}",
        ]
        if self.totals.get("pricing_enabled"):
            currency = self.totals["currency"]
            lines.append(
                f"Product cost  : {self.totals['product_cost']:,.2f} {currency}   "
                f"Application: {self.totals['application_cost']:,.2f} {currency}   "
                f"Total: {self.totals['total_cost']:,.2f} {currency}"
            )
        lines.append("")
        lines.append(f"{'zone':<22} {'rate':>10} {'area ha':>9} {'area %':>8} {'quantity':>12}")
        for row in self.rows:
            label = f"{row['class_code']} {row['relative_class']}"
            lines.append(
                f"{label:<22} {row['rate']:>10.4f} {row['area_ha']:>9.3f} "
                f"{row['area_percent']:>7.2f}% {row['product_quantity']:>12.4f}"
            )
        if self.warnings:
            lines.append("")
            lines.extend(f"WARNING: {warning}" for warning in self.warnings)
        if unit.strip().lower() in {"relative", "unitless", ""}:
            lines.append("")
            lines.append(
                "NOTE: rates are relative, not physical units. Set [prescription] unit and "
                "rate_by_class (or --unit/--rate) to get real quantities and costs."
            )
        lines.append("")
        lines.append(
            "NOTE: the rate schedule is an operator decision. This toolkit has no ground "
            "truth and does not validate that these rates are agronomically correct. "
            "Zones are equal-frequency, so each covers ~1/5 of the field by construction."
        )
        return "\n".join(lines)


def validate_schedule(
    schedule: RateSchedule, *, classes_present: Sequence[int]
) -> list[str]:
    """Check a schedule against the classes actually found in the raster.

    Returns non-fatal warnings; raises :class:`PrescriptionError` for anything
    that would silently produce a wrong map.
    """
    if not schedule.rates:
        raise PrescriptionError("The rate schedule is empty.")

    present = sorted(set(int(code) for code in classes_present))
    unknown = [code for code in present if code not in schedule.rates]
    if unknown:
        raise PrescriptionError(
            f"The class raster contains class code(s) {unknown} with no configured rate. "
            f"Configured classes: {sorted(schedule.rates)}."
        )

    negative = sorted(code for code, rate in schedule.rates.items() if rate < 0)
    if negative:
        raise PrescriptionError(f"Negative rates are not meaningful: classes {negative}.")

    warnings: list[str] = []
    ordered_codes = sorted(schedule.rates)
    ordered_rates = [schedule.rates[code] for code in ordered_codes]
    drops = [
        (ordered_codes[i], ordered_codes[i + 1])
        for i in range(len(ordered_rates) - 1)
        if ordered_rates[i] > ordered_rates[i + 1]
    ]
    if drops:
        pairs = ", ".join(f"{a}->{b}" for a, b in drops)
        warnings.append(
            f"The rate schedule decreases with vigour ({pairs}): the greener zone receives "
            f"less product. That is legitimate for some products and a mistake for others - "
            f"confirm it is intended."
        )
    return warnings


def _class_counts_and_area(
    class_raster: Path | str,
) -> tuple[dict[int, int], float, float]:
    """Count pixels per class and derive the pixel area, reading window by window."""
    counts: dict[int, int] = {}
    with rasterio.open(class_raster) as src:
        if src.count != 1:
            raise PrescriptionError(
                f"Expected a single-band class raster, got {src.count} bands: {class_raster}"
            )
        if src.dtypes[0] != "uint8":
            raise PrescriptionError(
                f"Expected a uint8 class raster (as written by `m3m zones`), "
                f"got {src.dtypes[0]}: {class_raster}"
            )
        if src.crs is None:
            raise PrescriptionError(
                f"The class raster has no CRS, so areas cannot be computed: {class_raster}"
            )
        pixel_area_m2 = abs(
            src.transform.a * src.transform.e - src.transform.b * src.transform.d
        )
        if pixel_area_m2 <= 0:
            raise PrescriptionError(f"The class raster has a degenerate transform: {class_raster}")
        for _, window in src.block_windows(1):
            block = src.read(1, window=window)
            codes, block_counts = np.unique(block, return_counts=True)
            for code, count in zip(codes.tolist(), block_counts.tolist(), strict=True):
                if code == 0:  # NoData
                    continue
                counts[int(code)] = counts.get(int(code), 0) + int(count)
    total = sum(counts.values())
    if total == 0:
        raise PrescriptionError(f"The class raster contains no valid pixels: {class_raster}")
    return counts, pixel_area_m2, float(total)


def _write_rate_raster(
    class_raster: Path | str,
    output_path: Path | str,
    schedule: RateSchedule,
    *,
    schedule_text: str,
    target_mean_rate: float | None,
) -> Path:
    """Map each class code to its rate and write the result as float32."""
    codes = sorted(schedule.rates)
    if not codes:
        raise PrescriptionError("The rate schedule is empty.")
    highest = max(codes)
    rate_lookup = np.zeros(highest + 1, dtype=np.float32)
    for code, rate in schedule.rates.items():
        rate_lookup[code] = rate

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(class_raster) as src:
        profile = src.profile.copy()
        profile.update(
            count=1,
            dtype="float32",
            nodata=NODATA_RATE,
            compress="deflate",
            predictor=3,
            tiled=True,
            blockxsize=min(512, src.width),
            blockysize=min(512, src.height),
            BIGTIFF="IF_SAFER",
        )
        with rasterio.open(output_path, "w", **profile) as dst:
            dst.set_band_description(1, f"Application rate ({schedule.unit})")
            dst.update_tags(
                1,
                PRODUCT="Variable-rate prescription",
                UNIT=schedule.unit,
                RATE_SCHEDULE=schedule_text,
                TARGET_MEAN_RATE="" if target_mean_rate is None else f"{target_mean_rate}",
                SOURCE_CLASS_RASTER=str(class_raster),
                ZONE_DEFINITION=(
                    "Relative vigour quintiles within this field and date; "
                    "equal-frequency, not agronomic thresholds"
                ),
                NOTE=(
                    "The rate schedule is an operator decision, not a model output. No ground "
                    "truth is used and no agronomic correctness is claimed."
                ),
            )
            for _, window in src.block_windows(1):
                block = src.read(1, window=window)
                known = (block > 0) & (block <= highest)
                unknown = block > highest
                if np.any(unknown):
                    raise PrescriptionError(
                        f"Class raster contains code(s) "
                        f"{sorted(set(block[unknown].tolist()))} with no configured rate."
                    )
                out = np.full(block.shape, NODATA_RATE, dtype=np.float32)
                out[known] = rate_lookup[block[known]]
                dst.write(out, 1, window=window)
    return output_path


def build(
    class_raster: Path | str,
    *,
    output_dir: Path | str,
    prefix: str = "field",
    index_name: str = "ndvi",
    schedule: RateSchedule,
    target_mean_rate: float | None = None,
    pricing: Pricing | None = None,
    boundary: Path | str | None = None,
    make_figure: bool = True,
) -> PrescriptionResult:
    """Build a prescription map, its summary table and its totals.

    ``target_mean_rate`` rescales the schedule so the area-weighted mean rate
    equals the requested value, which keeps the total product under operator
    control even though the zones themselves are equal-frequency.
    """
    class_raster = Path(class_raster)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pricing = pricing or Pricing()
    index_name = index_name.lower()

    counts, pixel_area_m2, valid_pixels = _class_counts_and_area(class_raster)
    warnings = validate_schedule(schedule, classes_present=list(counts))

    if target_mean_rate is not None:
        current_mean = sum(
            count * schedule.rate_for(code) for code, count in counts.items()
        ) / valid_pixels
        if current_mean <= 0:
            raise PrescriptionError(
                "Cannot rescale to a target mean because the current schedule has a "
                "zero area-weighted mean. Give the zones non-zero relative weights."
            )
        schedule = schedule.rescaled(target_mean_rate / current_mean)
        if max(schedule.rates.values()) <= 0:
            raise PrescriptionError("Rescaling produced a non-positive schedule.")

    schedule_text = "; ".join(
        f"{code}={schedule.rates[code]:g}" for code in sorted(schedule.rates)
    )

    rate_raster = output_dir / f"{prefix}_{index_name}_prescription.tif"
    _write_rate_raster(
        class_raster,
        rate_raster,
        schedule,
        schedule_text=schedule_text,
        target_mean_rate=target_mean_rate,
    )

    rows: list[dict[str, Any]] = []
    quantity_unit_text = quantity_unit(schedule.unit)
    for code in sorted(counts):
        count = counts[code]
        rate = schedule.rate_for(code)
        area_ha = count * pixel_area_m2 / 10_000.0
        quantity = rate * area_ha
        product_cost = quantity * pricing.product_price_per_unit
        application_cost = area_ha * pricing.application_cost_per_ha
        rows.append(
            {
                "class_code": code,
                "relative_class": (
                    RELATIVE_CLASS_NAMES[code - 1]
                    if 1 <= code <= len(RELATIVE_CLASS_NAMES)
                    else f"class {code}"
                ),
                "rate": rate,
                "unit": schedule.unit,
                "pixel_count": count,
                "area_ha": area_ha,
                "area_percent": 100.0 * count / valid_pixels,
                "product_quantity": quantity,
                "quantity_unit": quantity_unit_text,
                "product_cost": product_cost,
                "application_cost": application_cost,
                "total_cost": product_cost + application_cost,
            }
        )

    summary_csv = output_dir / f"{prefix}_{index_name}_prescription_summary.csv"
    with summary_csv.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(SUMMARY_FIELDS))
        writer.writeheader()
        writer.writerows(rows)

    area_ha = valid_pixels * pixel_area_m2 / 10_000.0
    quantity_total = sum(row["product_quantity"] for row in rows)
    totals: dict[str, Any] = {
        "source_class_raster": str(class_raster),
        "rate_raster": str(rate_raster),
        "index": index_name.upper(),
        "unit": schedule.unit,
        "quantity_unit": quantity_unit_text,
        "rate_schedule": {str(code): rate for code, rate in sorted(schedule.rates.items())},
        "rate_schedule_text": schedule_text,
        "target_mean_rate": target_mean_rate,
        "mean_rate": quantity_total / area_ha if area_ha else 0.0,
        "zone_count": len(rows),
        "valid_pixels": int(valid_pixels),
        "pixel_area_m2": pixel_area_m2,
        "area_ha": area_ha,
        "product_quantity": quantity_total,
        "pricing_enabled": pricing.enabled,
        "currency": pricing.currency,
        "product_price_per_unit": pricing.product_price_per_unit,
        "application_cost_per_ha": pricing.application_cost_per_ha,
        "product_cost": sum(row["product_cost"] for row in rows),
        "application_cost": sum(row["application_cost"] for row in rows),
        "total_cost": sum(row["total_cost"] for row in rows),
        "zones": rows,
        "warnings": warnings,
        "limitations": [
            "The rate schedule is an operator decision; no ground truth is used to derive "
            "or validate it.",
            "Zones are within-field equal-frequency quantiles, not agronomic thresholds.",
            "Areas are pixel counts times the raster's nominal pixel area and ignore "
            "headland, overlap and implement turning losses.",
        ],
    }
    totals_json = output_dir / f"{prefix}_{index_name}_prescription_totals.json"
    totals_json.write_text(
        json.dumps(totals, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    figure: Path | None = None
    if make_figure:
        figure = create_figure(
            rate_raster,
            rows,
            output_dir / f"{prefix}_{index_name}_prescription.png",
            boundary=boundary,
            unit=schedule.unit,
            quantity_unit_label=quantity_unit_text,
            index_name=index_name.upper(),
        )

    return PrescriptionResult(
        rate_raster=rate_raster,
        summary_csv=summary_csv,
        totals_json=totals_json,
        figure=figure,
        rows=rows,
        totals=totals,
        warnings=warnings,
    )


def create_figure(
    rate_raster: Path | str,
    rows: Sequence[Mapping[str, Any]],
    output_path: Path | str,
    *,
    boundary: Path | str | None = None,
    unit: str = "relative",
    quantity_unit_label: str | None = None,
    index_name: str = "NDVI",
    downsample: int = 8,
) -> Path:
    """Write the field-operation figure: the rate map beside a per-zone quantity chart."""
    output_path = Path(output_path)
    distinct = sorted({float(row["rate"]) for row in rows})
    colours = ListedColormap([ZONE_COLORS[i % len(ZONE_COLORS)] for i in range(len(distinct))])
    if len(distinct) > 1:
        step = min(b - a for a, b in zip(distinct, distinct[1:], strict=False)) / 2.0
    else:
        step = max(abs(distinct[0]) * 0.1, 0.5)
    norm = BoundaryNorm(
        [distinct[0] - step, *[value + step for value in distinct]], colours.N
    )

    figure, (map_ax, bar_ax) = plt.subplots(
        1, 2, figsize=(13, 5.6), constrained_layout=True, gridspec_kw={"width_ratios": [1.35, 1]}
    )

    with rasterio.open(rate_raster) as src:
        out_shape = (max(1, src.height // downsample), max(1, src.width // downsample))
        image = src.read(1, out_shape=out_shape, masked=True, resampling=Resampling.nearest)
        extent = (src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top)
        crs = src.crs

    map_ax.imshow(
        image, extent=extent, origin="upper", cmap=colours, norm=norm, interpolation="nearest"
    )
    if boundary is not None:
        gpd.read_file(boundary).to_crs(crs).boundary.plot(
            ax=map_ax, color="#202020", linewidth=0.8
        )
    map_ax.set_title(f"{index_name} prescription map", fontsize=11)
    map_ax.set_xlabel("Easting (m)", fontsize=8.5)
    map_ax.set_ylabel("Northing (m)", fontsize=8.5)
    map_ax.tick_params(labelsize=7.5)
    map_ax.set_aspect("equal")

    mappable = plt.cm.ScalarMappable(norm=norm, cmap=colours)
    colorbar = figure.colorbar(
        mappable,
        ax=map_ax,
        fraction=0.046,
        pad=0.02,
        ticks=distinct,
        boundaries=[distinct[0] - step, *[value + step for value in distinct]],
    )
    colorbar.ax.set_yticklabels([f"{value:g}" for value in distinct])
    colorbar.set_label(f"Application rate ({unit})", fontsize=8.5)
    colorbar.ax.tick_params(labelsize=7.5)

    labels = [
        f"{row['class_code']} {row['relative_class']}\n"
        f"{row['rate']:g} {unit} | {row['area_ha']:.2f} ha"
        for row in rows
    ]
    quantities = [float(row["product_quantity"]) for row in rows]
    positions = np.arange(len(rows))
    bar_ax.barh(
        positions,
        quantities,
        color=[ZONE_COLORS[(row["class_code"] - 1) % len(ZONE_COLORS)] for row in rows],
        edgecolor="#202020",
        linewidth=0.6,
    )
    bar_ax.set_yticks(positions, labels, fontsize=8)
    bar_ax.invert_yaxis()
    bar_ax.set_xlabel(
        f"Product quantity ({quantity_unit_label or quantity_unit(unit)})", fontsize=8.5
    )
    bar_ax.set_title("Product distributed per zone", fontsize=11)
    bar_ax.tick_params(axis="x", labelsize=7.5)
    bar_ax.grid(axis="x", alpha=0.25, linewidth=0.6)
    bar_ax.set_axisbelow(True)
    total = sum(quantities)
    for position, quantity in zip(positions, quantities, strict=True):
        share = 100.0 * quantity / total if total else 0.0
        bar_ax.text(
            quantity,
            position,
            f"  {share:.1f}%",
            va="center",
            fontsize=7.5,
        )

    figure.suptitle(
        "Variable-rate prescription\n"
        "Zones are within-field equal-frequency ranks; the rate schedule is an operator "
        "decision, not a validated agronomic recommendation",
        fontsize=10.5,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)
    return output_path
