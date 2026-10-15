"""End-to-end run of the whole pipeline on synthetic data.

This is the test that answers "does a fresh clone actually work?". It drives the
CLI exactly as a new user would and asserts on the artefacts that land on disk.
"""

from __future__ import annotations

from pathlib import Path

import rasterio

from m3m.cli import EXIT_OK, main
from m3m.constants import NODATA


def _run(config: Path, *args: str) -> int:
    return main([*args, "--config", str(config)])


def test_pipeline_runs_from_the_command_line(project_dir, capsys) -> None:
    project, config = project_dir

    assert _run(config, "audit") == EXIT_OK
    assert (project / "outputs" / "dataset_inventory.csv").is_file()

    assert _run(config, "panel") == EXIT_OK

    assert _run(config, "calibrate") == EXIT_OK
    radiometric_dir = project / "outputs" / "radiometric"
    assert (radiometric_dir / "radiometric_manifest.csv").is_file()
    calibrated = sorted(radiometric_dir.glob("*_REFLECTANCE_APPROX.tif"))
    assert len(calibrated) == 8  # 2 groups x 4 bands

    assert _run(config, "indices", "--raster", str(project / "synthetic_reflectance_orthomosaic.tif")) == EXIT_OK
    index_dir = project / "outputs" / "indices"
    for name in ("ndvi", "ndre", "evi2"):
        assert (index_dir / f"field_{name}.tif").is_file()
    assert (index_dir / "vegetation_index_summary.csv").is_file()

    assert _run(config, "zones") == EXIT_OK
    assert (index_dir / "field_ndvi_relative_5class.tif").is_file()
    assert (index_dir / "vegetation_index_relative_classes.png").is_file()

    assert _run(config, "compare") == EXIT_OK
    assert (index_dir / "vegetation_index_pairwise_comparison.csv").is_file()

    assert _run(config, "sensitivity") == EXIT_OK
    sensitivity_dir = project / "outputs" / "sensitivity"
    assert (sensitivity_dir / "proxy_weight_sensitivity.csv").is_file()
    assert (sensitivity_dir / "proxy_composite_quintile_maps.png").is_file()

    captured = capsys.readouterr()
    assert "error" not in captured.err.lower()


def test_calibration_is_resumable_and_idempotent(project_dir) -> None:
    project, config = project_dir
    assert _run(config, "calibrate") == EXIT_OK
    manifest = project / "outputs" / "radiometric" / "radiometric_manifest.csv"
    after_first = manifest.read_text(encoding="utf-8-sig").count("ok")

    assert _run(config, "calibrate") == EXIT_OK
    # Existing outputs are skipped, so the manifest does not grow duplicates.
    assert manifest.read_text(encoding="utf-8-sig").count("ok") == after_first


def test_calibrated_output_is_physically_plausible(project_dir) -> None:
    project, config = project_dir
    assert _run(config, "calibrate") == EXIT_OK
    sample = next((project / "outputs" / "radiometric").glob("*_MS_NIR*_REFLECTANCE_APPROX.tif"))
    with rasterio.open(sample) as src:
        values = src.read(1)
        assert src.nodata == NODATA
    assert values.min() >= 0.0
    assert values.max() <= 1.0


def test_sensitivity_reports_correlations_within_bounds(project_dir) -> None:
    import csv

    project, config = project_dir
    raster = project / "synthetic_reflectance_orthomosaic.tif"
    assert _run(config, "indices", "--raster", str(raster)) == EXIT_OK
    assert _run(config, "sensitivity") == EXIT_OK

    path = project / "outputs" / "sensitivity" / "proxy_weight_sensitivity.csv"
    with path.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    assert rows
    for row in rows:
        rho = float(row["spearman_rho_histogram_midranks"])
        assert -1.0 <= rho <= 1.0
        assert 0.0 <= float(row["same_quintile_percent"]) <= 100.0


def test_prefix_is_configurable(project_dir) -> None:
    project, config = project_dir
    assert _run(config, "indices", "--raster", str(project / "synthetic_reflectance_orthomosaic.tif"), "--prefix", "winterbarley") == EXIT_OK
    assert (project / "outputs" / "indices" / "winterbarley_ndvi.tif").is_file()


def test_out_dir_override_relocates_everything(tmp_path: Path, project_dir) -> None:
    project, config = project_dir
    other = tmp_path / "elsewhere"
    assert main(
        [
            "indices",
            "--config",
            str(config),
            "--raster",
            str(project / "synthetic_reflectance_orthomosaic.tif"),
            "--out-dir",
            str(other),
        ]
    ) == EXIT_OK
    assert (other / "indices" / "field_ndvi.tif").is_file()


def test_assess_flags_a_corrupted_orthomosaic(project_dir, tmp_path: Path) -> None:
    import json

    project, config = project_dir
    damaged = tmp_path / "damaged.tif"
    with rasterio.open(project / "synthetic_reflectance_orthomosaic.tif") as src:
        profile = src.profile.copy()
        data = src.read()
        descriptions = src.descriptions
    # Damage pixels well inside the field boundary (it is inset from the edges)
    # and keep the band descriptions, which `assess` validates.
    data[1, 40, 60] = -5.0  # Green: physically impossible negative reflectance
    data[2, 40, 61] = 2.0  # NIR: above unity
    with rasterio.open(damaged, "w", **profile) as dst:
        dst.write(data)
        for position, name in enumerate(descriptions, start=1):
            if name:
                dst.set_band_description(position, name)

    assert main(["assess", "--config", str(config), "--raster", str(damaged)]) == EXIT_OK
    report = json.loads(
        (project / "outputs" / "orthomosaic_quality.json").read_text(encoding="utf-8")
    )
    assert report["bands"]["Green"]["less_than_zero"] >= 1
    assert report["bands"]["NIR"]["greater_than_one"] >= 1
    assert report["boundary_coverage_fraction"] > 0
