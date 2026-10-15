"""The prescription map: rate mapping, rescaling, totals and cost arithmetic."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest
import rasterio

from m3m import indices, prescription, zones
from m3m.config import ConfigError, load_settings
from m3m.prescription import (
    NODATA_RATE,
    PrescriptionError,
    Pricing,
    RateSchedule,
    build,
    quantity_unit,
    validate_schedule,
)


@pytest.fixture()
def class_raster(tmp_path: Path, orthomosaic) -> Path:
    """A real relative-class raster produced by the normal pipeline."""
    index_dir = tmp_path / "indices"
    indices.calculate(orthomosaic.raster, orthomosaic.boundary, index_dir, prefix="rx")
    zones.run(index_dir, orthomosaic.boundary, prefix="rx", make_figures=False)
    return index_dir / "rx_ndvi_relative_5class.tif"


# --------------------------------------------------------------------------- #
# pure helpers
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("unit", "expected"),
    [
        ("kg N/ha", "kg N"),
        ("L/ha", "L"),
        ("kg/ha", "kg"),
        ("seeds per ha", "seeds"),
        ("relative", "relative*ha"),
        ("unitless", "relative*ha"),
        ("kg", "kg*ha"),
    ],
)
def test_quantity_unit_cancels_the_per_hectare_divisor(unit: str, expected: str) -> None:
    assert quantity_unit(unit) == expected


def test_schedule_rescaling_is_proportional() -> None:
    schedule = RateSchedule({1: 0.0, 2: 20.0, 3: 40.0}, unit="kg/ha")
    scaled = schedule.rescaled(2.5)
    assert scaled.rates == {1: 0.0, 2: 50.0, 3: 100.0}
    assert scaled.unit == "kg/ha"
    assert schedule.rates[2] == 20.0  # the original is untouched


def test_unknown_class_in_the_raster_is_an_error() -> None:
    with pytest.raises(PrescriptionError, match="no configured rate"):
        validate_schedule(RateSchedule({1: 0.0, 2: 5.0}), classes_present=[1, 2, 3])


def test_negative_rate_is_rejected() -> None:
    with pytest.raises(PrescriptionError, match="Negative rates"):
        validate_schedule(RateSchedule({1: -1.0, 2: 5.0}), classes_present=[1, 2])


def test_empty_schedule_is_rejected() -> None:
    with pytest.raises(PrescriptionError, match="empty"):
        validate_schedule(RateSchedule({}), classes_present=[1])


def test_decreasing_schedule_warns_but_does_not_fail() -> None:
    warnings = validate_schedule(
        RateSchedule({1: 100.0, 2: 50.0, 3: 0.0}), classes_present=[1, 2, 3]
    )
    assert len(warnings) == 1
    assert "decreases with vigour" in warnings[0]
    assert "1->2" in warnings[0]


def test_increasing_schedule_produces_no_warning() -> None:
    assert validate_schedule(RateSchedule({1: 0.0, 2: 5.0}), classes_present=[1, 2]) == []


# --------------------------------------------------------------------------- #
# building
# --------------------------------------------------------------------------- #


def test_rate_raster_maps_every_class_to_its_rate(
    tmp_path: Path, class_raster: Path, orthomosaic
) -> None:
    schedule = RateSchedule({1: 0.0, 2: 40.0, 3: 80.0, 4: 120.0, 5: 160.0}, unit="kg N/ha")
    result = build(
        class_raster,
        output_dir=tmp_path / "rx",
        prefix="rx",
        schedule=schedule,
        boundary=orthomosaic.boundary,
    )

    with rasterio.open(class_raster) as src:
        classes = src.read(1)
    with rasterio.open(result.rate_raster) as src:
        rates = src.read(1)
        assert src.nodata == NODATA_RATE
        tags = src.tags(1)

    for code, rate in schedule.rates.items():
        assert np.all(rates[classes == code] == pytest.approx(rate))
    # NoData stays NoData; it must never be mapped to a rate of zero.
    assert np.all(rates[classes == 0] == NODATA_RATE)
    assert tags["UNIT"] == "kg N/ha"
    assert "operator decision" in tags["NOTE"]


def test_every_zone_gets_one_row_and_areas_sum_to_the_field(
    tmp_path: Path, class_raster: Path
) -> None:
    result = build(
        class_raster,
        output_dir=tmp_path / "rx",
        prefix="rx",
        schedule=RateSchedule({1: 0.0, 2: 25.0, 3: 50.0, 4: 75.0, 5: 100.0}, unit="kg/ha"),
        make_figure=False,
    )
    assert len(result.rows) == 5
    assert [row["class_code"] for row in result.rows] == [1, 2, 3, 4, 5]
    assert sum(row["area_ha"] for row in result.rows) == pytest.approx(
        result.totals["area_ha"]
    )
    assert sum(row["pixel_count"] for row in result.rows) == result.totals["valid_pixels"]
    # The zones are equal-frequency, so the split is ~20% each by construction.
    for row in result.rows:
        assert row["area_percent"] == pytest.approx(20.0, abs=1.0)


def test_target_mean_rate_is_essentially_reached(tmp_path: Path, class_raster: Path) -> None:
    """The rounded schedule must still land on the requested field average."""
    result = build(
        class_raster,
        output_dir=tmp_path / "rx",
        prefix="rx",
        schedule=RateSchedule({1: 0.0, 2: 40.0, 3: 80.0, 4: 120.0, 5: 160.0}, unit="kg N/ha"),
        target_mean_rate=100.0,
        make_figure=False,
    )
    assert result.totals["mean_rate"] == pytest.approx(100.0, rel=1e-3)
    # The area-weighted mean is what the raster must actually contain, measured
    # from the written pixels rather than from the schedule.
    with rasterio.open(result.rate_raster) as src:
        rates = src.read(1)
    valid = rates != NODATA_RATE
    assert float(rates[valid].mean()) == pytest.approx(100.0, rel=1e-3)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (239.9958, 240.0),  # the real output of a 120 kg/ha target
        (179.99685, 180.0),
        (119.9979, 120.0),
        (59.99895, 60.0),
        (0.00399993, 0.004),  # tiny unit magnitudes must survive rounding
        (0.0, 0.0),
        (-12.34567, -12.35),
    ],
)
def test_round_to_significant_keeps_four_figures(value: float, expected: float) -> None:
    assert prescription.round_to_significant(value) == pytest.approx(expected)


def test_rescaled_rates_are_controller_usable(tmp_path: Path, class_raster: Path) -> None:
    """A realistic 120 kg N/ha target must not yield numbers like 239.996."""
    result = build(
        class_raster,
        output_dir=tmp_path / "rx",
        prefix="rx",
        schedule=RateSchedule({1: 0.0, 2: 40.0, 3: 80.0, 4: 120.0, 5: 160.0}, unit="kg N/ha"),
        target_mean_rate=120.0,
        make_figure=False,
    )
    for rate in result.totals["rate_schedule"].values():
        # Every rate is its own 4-significant-figure rounding.
        assert prescription.round_to_significant(rate) == pytest.approx(rate)
    assert result.totals["rate_schedule"]["5"] == pytest.approx(240.0, abs=0.01)


def test_rescaling_keeps_the_shape_of_the_schedule(tmp_path: Path, class_raster: Path) -> None:
    plain = build(
        class_raster,
        output_dir=tmp_path / "a",
        prefix="rx",
        schedule=RateSchedule({1: 0.0, 2: 10.0, 3: 20.0, 4: 30.0, 5: 40.0}, unit="kg/ha"),
        make_figure=False,
    )
    scaled = build(
        class_raster,
        output_dir=tmp_path / "b",
        prefix="rx",
        schedule=RateSchedule({1: 0.0, 2: 10.0, 3: 20.0, 4: 30.0, 5: 40.0}, unit="kg/ha"),
        target_mean_rate=200.0,
        make_figure=False,
    )
    factor = scaled.totals["rate_schedule"]["5"] / plain.totals["rate_schedule"]["5"]
    for code in ("2", "3", "4", "5"):
        assert scaled.totals["rate_schedule"][code] == pytest.approx(
            plain.totals["rate_schedule"][code] * factor
        )


def test_zero_schedule_cannot_be_rescaled(tmp_path: Path, class_raster: Path) -> None:
    with pytest.raises(PrescriptionError, match="zero area-weighted mean"):
        build(
            class_raster,
            output_dir=tmp_path / "rx",
            prefix="rx",
            schedule=RateSchedule({1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0}, unit="kg/ha"),
            target_mean_rate=100.0,
            make_figure=False,
        )


def test_cost_arithmetic(tmp_path: Path, class_raster: Path) -> None:
    result = build(
        class_raster,
        output_dir=tmp_path / "rx",
        prefix="rx",
        schedule=RateSchedule({1: 0.0, 2: 50.0, 3: 100.0, 4: 150.0, 5: 200.0}, unit="kg N/ha"),
        pricing=Pricing(
            product_price_per_unit=4.5, currency="CNY", application_cost_per_ha=25.0
        ),
        make_figure=False,
    )
    totals = result.totals
    assert totals["pricing_enabled"] is True
    assert totals["product_cost"] == pytest.approx(
        totals["product_quantity"] * 4.5, rel=1e-9
    )
    assert totals["application_cost"] == pytest.approx(totals["area_ha"] * 25.0, rel=1e-9)
    assert totals["total_cost"] == pytest.approx(
        totals["product_cost"] + totals["application_cost"], rel=1e-9
    )
    for row in result.rows:
        assert row["total_cost"] == pytest.approx(
            row["product_cost"] + row["application_cost"]
        )


def test_pricing_disabled_by_default(tmp_path: Path, class_raster: Path) -> None:
    result = build(
        class_raster,
        output_dir=tmp_path / "rx",
        prefix="rx",
        schedule=RateSchedule({1: 0.0, 2: 1.0, 3: 2.0, 4: 3.0, 5: 4.0}),
        make_figure=False,
    )
    assert result.totals["pricing_enabled"] is False
    assert "Product cost" not in result.format_report()


def test_summary_csv_and_totals_json_are_written(tmp_path: Path, class_raster: Path) -> None:
    result = build(
        class_raster,
        output_dir=tmp_path / "rx",
        prefix="rx",
        index_name="ndvi",
        schedule=RateSchedule({1: 0.0, 2: 40.0, 3: 80.0, 4: 120.0, 5: 160.0}, unit="kg N/ha"),
        make_figure=False,
    )
    with result.summary_csv.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 5
    assert list(rows[0]) == list(prescription.SUMMARY_FIELDS)
    assert rows[0]["quantity_unit"] == "kg N"

    payload = json.loads(result.totals_json.read_text(encoding="utf-8"))
    assert payload["quantity_unit"] == "kg N"
    assert payload["rate_schedule"]["5"] == 160.0
    assert len(payload["limitations"]) >= 3
    assert payload["zones"][0]["class_code"] == 1


def test_report_states_the_operator_decision_and_the_equal_frequency_caveat(
    tmp_path: Path, class_raster: Path
) -> None:
    result = build(
        class_raster,
        output_dir=tmp_path / "rx",
        prefix="rx",
        schedule=RateSchedule({1: 0.0, 2: 0.25, 3: 0.5, 4: 0.75, 5: 1.0}),
        make_figure=False,
    )
    report = result.format_report()
    assert "operator decision" in report
    assert "no ground truth" in report
    assert "equal-frequency" in report
    # The default unit is relative, so the report must say so rather than
    # letting the numbers pass as physical quantities.
    assert "rates are relative, not physical units" in report


def test_figure_is_written(tmp_path: Path, class_raster: Path, orthomosaic) -> None:
    result = build(
        class_raster,
        output_dir=tmp_path / "rx",
        prefix="rx",
        schedule=RateSchedule({1: 0.0, 2: 40.0, 3: 80.0, 4: 120.0, 5: 160.0}, unit="kg N/ha"),
        boundary=orthomosaic.boundary,
    )
    assert result.figure is not None
    assert result.figure.is_file()
    assert result.figure.stat().st_size > 5_000


def test_figure_works_without_a_boundary(tmp_path: Path, class_raster: Path) -> None:
    result = build(
        class_raster,
        output_dir=tmp_path / "rx",
        prefix="rx",
        schedule=RateSchedule({1: 0.0, 2: 1.0, 3: 2.0, 4: 3.0, 5: 4.0}),
        boundary=None,
    )
    assert result.figure is not None and result.figure.is_file()


def test_equal_rates_for_every_zone_still_works(tmp_path: Path, class_raster: Path) -> None:
    """A flat schedule is a valid 'no variation' case and must not crash the figure."""
    result = build(
        class_raster,
        output_dir=tmp_path / "rx",
        prefix="rx",
        schedule=RateSchedule({1: 5.0, 2: 5.0, 3: 5.0, 4: 5.0, 5: 5.0}, unit="kg/ha"),
    )
    assert result.totals["mean_rate"] == pytest.approx(5.0)
    assert result.figure is not None and result.figure.is_file()


def test_non_uint8_input_is_rejected(tmp_path: Path, orthomosaic) -> None:
    index_dir = tmp_path / "indices"
    indices.calculate(orthomosaic.raster, orthomosaic.boundary, index_dir, prefix="rx")
    continuous = index_dir / "rx_ndvi.tif"
    with pytest.raises(PrescriptionError, match="uint8 class raster"):
        build(
            continuous,
            output_dir=tmp_path / "rx",
            prefix="rx",
            schedule=RateSchedule({1: 0.0, 2: 1.0, 3: 2.0, 4: 3.0, 5: 4.0}),
            make_figure=False,
        )


def test_unconfigured_class_code_is_rejected(tmp_path: Path, class_raster: Path) -> None:
    with pytest.raises(PrescriptionError, match="no configured rate"):
        build(
            class_raster,
            output_dir=tmp_path / "rx",
            prefix="rx",
            schedule=RateSchedule({1: 0.0, 2: 1.0}),  # classes 3-5 are missing
            make_figure=False,
        )


# --------------------------------------------------------------------------- #
# configuration
# --------------------------------------------------------------------------- #


def test_prescription_defaults_are_relative_not_physical(tmp_path: Path) -> None:
    settings = load_settings(env={}, cwd=tmp_path)
    assert settings.prescription.unit == "relative"
    assert settings.prescription.target_mean_rate is None
    assert settings.prescription.product_price_per_unit == 0.0
    assert sorted(settings.prescription.rate_by_class) == [1, 2, 3, 4, 5]


def test_prescription_read_from_toml(tmp_path: Path) -> None:
    config = tmp_path / "m3m.toml"
    config.write_text(
        "[prescription]\n"
        'unit = "kg N/ha"\n'
        "target_mean_rate = 120.0\n"
        "product_price_per_unit = 4.5\n"
        "application_cost_per_ha = 25.0\n"
        'currency = "AUD"\n'
        "[prescription.rate_by_class]\n"
        "1 = 0\n2 = 40\n3 = 80\n4 = 120\n5 = 160\n",
        encoding="utf-8",
    )
    settings = load_settings(config, env={}, cwd=tmp_path)
    assert settings.prescription.unit == "kg N/ha"
    assert settings.prescription.target_mean_rate == 120.0
    assert settings.prescription.rate_by_class == {1: 0.0, 2: 40.0, 3: 80.0, 4: 120.0, 5: 160.0}
    assert settings.prescription.currency == "AUD"
    assert settings.sources["prescription.rate_by_class"] == "config file"


def test_prescription_cli_overrides_toml(tmp_path: Path) -> None:
    config = tmp_path / "m3m.toml"
    config.write_text(
        '[prescription]\nunit = "kg/ha"\n'
        "[prescription.rate_by_class]\n1 = 0\n2 = 10\n3 = 20\n4 = 30\n5 = 40\n",
        encoding="utf-8",
    )
    settings = load_settings(
        config,
        {"prescription": {"unit": "L/ha", "rate_by_class": {4: 99.0}}},
        env={},
        cwd=tmp_path,
    )
    assert settings.prescription.unit == "L/ha"
    # `--rate` merges: the flag only has to name the classes the operator wants
    # to change, and everything else keeps whatever the config file said.
    assert settings.prescription.rate_by_class == {
        1: 0.0,
        2: 10.0,
        3: 20.0,
        4: 99.0,
        5: 40.0,
    }
    assert settings.sources["prescription.rate_by_class"] == "command line"


def test_cli_rate_can_replace_every_class_at_once(tmp_path: Path) -> None:
    """Naming all five classes behaves like a full replacement."""
    settings = load_settings(
        None,
        {"prescription": {"rate_by_class": {1: 0.0, 2: 1.0, 3: 2.0, 4: 3.0, 5: 4.0}}},
        env={},
        cwd=tmp_path,
    )
    assert settings.prescription.rate_by_class == {1: 0.0, 2: 1.0, 3: 2.0, 4: 3.0, 5: 4.0}


def test_cli_rate_merges_over_the_builtin_defaults(tmp_path: Path) -> None:
    settings = load_settings(
        None, {"prescription": {"rate_by_class": {2: 2.0}}}, env={}, cwd=tmp_path
    )
    rates = settings.prescription.rate_by_class
    assert rates[2] == 2.0
    assert sorted(rates) == [1, 2, 3, 4, 5]


def test_prescription_environment_variable(tmp_path: Path) -> None:
    settings = load_settings(
        None, env={"M3M_PRESCRIPTION_UNIT": "L/ha"}, cwd=tmp_path
    )
    assert settings.prescription.unit == "L/ha"


@pytest.mark.parametrize("bad", ["[1.5, 2.0]", '"kg/ha"'])
def test_non_table_rate_schedule_is_rejected(tmp_path: Path, bad: str) -> None:
    config = tmp_path / "m3m.toml"
    config.write_text(f"[prescription]\nrate_by_class = {bad}\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="rate_by_class"):
        load_settings(config, env={}, cwd=tmp_path)


def test_class_zero_is_rejected(tmp_path: Path) -> None:
    config = tmp_path / "m3m.toml"
    config.write_text("[prescription.rate_by_class]\n0 = 10\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="start at 1"):
        load_settings(config, env={}, cwd=tmp_path)


def test_non_integer_class_key_is_rejected(tmp_path: Path) -> None:
    config = tmp_path / "m3m.toml"
    config.write_text('[prescription.rate_by_class]\n"low" = 10\n', encoding="utf-8")
    with pytest.raises(ConfigError, match="integer class codes"):
        load_settings(config, env={}, cwd=tmp_path)


def test_negative_target_mean_rate_is_rejected(tmp_path: Path) -> None:
    config = tmp_path / "m3m.toml"
    config.write_text("[prescription]\ntarget_mean_rate = -5\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="must be positive"):
        load_settings(config, env={}, cwd=tmp_path)


def test_empty_unit_is_rejected(tmp_path: Path) -> None:
    config = tmp_path / "m3m.toml"
    config.write_text('[prescription]\nunit = "  "\n', encoding="utf-8")
    with pytest.raises(ConfigError, match="must not be empty"):
        load_settings(config, env={}, cwd=tmp_path)
