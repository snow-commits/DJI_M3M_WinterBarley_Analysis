"""CLI behaviour: exit codes, help, and the quality of the error messages."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from m3m import __version__
from m3m.cli import EXIT_OK, EXIT_USAGE, main


def test_version_flag_reports_the_package_version(capsys) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["--version"])
    assert excinfo.value.code == 0
    assert __version__ in capsys.readouterr().out


def test_no_command_prints_help_and_fails(capsys) -> None:
    assert main([]) == EXIT_USAGE
    assert "usage:" in capsys.readouterr().out


def test_unknown_command_is_a_usage_error(capsys) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["frobnicate"])
    assert excinfo.value.code == EXIT_USAGE


def test_config_show_works_without_any_configuration(tmp_path: Path, monkeypatch, capsys) -> None:
    monkeypatch.chdir(tmp_path)
    assert main(["config", "show"]) == EXIT_OK
    payload = json.loads(capsys.readouterr().out)
    assert payload["config_file"] is None
    assert payload["paths"]["img_dir"] is None
    assert payload["output"]["prefix"] == "field"


def test_config_init_then_refuses_to_clobber(tmp_path: Path, capsys) -> None:
    target = tmp_path / "m3m.toml"
    assert main(["config", "init", "--path", str(target)]) == EXIT_OK
    assert target.is_file()
    assert "[radiometric.panel_roi]" in target.read_text(encoding="utf-8")

    assert main(["config", "init", "--path", str(target)]) == EXIT_USAGE
    assert "already exists" in capsys.readouterr().err

    assert main(["config", "init", "--path", str(target), "--force"]) == EXIT_OK


def test_missing_input_path_produces_an_actionable_message(tmp_path: Path, monkeypatch, capsys) -> None:
    monkeypatch.chdir(tmp_path)
    assert main(["audit"]) == EXIT_USAGE
    err = capsys.readouterr().err
    assert "--img-dir" in err
    assert "M3M_IMG_DIR" in err
    assert "m3m.example.toml" in err


def test_config_flag_is_accepted_before_and_after_the_subcommand(project_dir, capsys) -> None:
    project, config = project_dir
    assert main(["--config", str(config), "audit"]) == EXIT_OK
    assert main(["audit", "--config", str(config)]) == EXIT_OK
    assert "Capture groups" in capsys.readouterr().out


def test_command_line_overrides_the_config_file(project_dir, tmp_path: Path) -> None:
    project, config = project_dir
    other = tmp_path / "cli_out"
    assert main(["audit", "--config", str(config), "--out-dir", str(other)]) == EXIT_OK
    assert (other / "dataset_inventory.csv").is_file()


def test_env_var_is_used_when_no_flag_is_given(project_dir, monkeypatch) -> None:
    project, config = project_dir
    monkeypatch.setenv("M3M_OUT_DIR", str(project / "env_out"))
    assert main(["audit", "--config", str(config)]) == EXIT_OK
    assert (project / "env_out" / "dataset_inventory.csv").is_file()


def test_malformed_roi_flag_is_rejected(project_dir, capsys) -> None:
    project, config = project_dir
    assert main(["panel", "--config", str(config), "--roi-band", "NIR=1,2,3"]) == EXIT_USAGE
    assert "four integers" in capsys.readouterr().err


def test_orthorectify_renders_a_script_without_docker(project_dir, capsys) -> None:
    project, config = project_dir
    assert main(["calibrate", "--config", str(config)]) == EXIT_OK
    assert main(["orthorectify", "--config", str(config)]) == EXIT_OK

    script = project / "outputs" / "orthorectify.sh"
    assert script.is_file()
    text = script.read_text(encoding="utf-8")
    # Rendered from configuration, so no machine-specific path may survive.
    assert "@@" not in text
    assert "/mnt/e/" not in text
    assert "opendronemap/odm" in text
    assert "-m m3m fill-nodata" in text
    assert "\r\n" not in text


def test_orthorectify_requires_a_prior_calibration(project_dir, capsys) -> None:
    project, config = project_dir
    assert main(["orthorectify", "--config", str(config)]) == EXIT_USAGE
    assert "Run `m3m calibrate` first" in capsys.readouterr().err
