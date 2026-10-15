"""Configuration resolution: precedence, path handling and error messages."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from m3m.config import (
    DEFAULT_OUT_DIR,
    DEFAULT_SATURATION_DN,
    ConfigError,
    find_config_file,
    load_settings,
)


def test_no_source_leaves_inputs_unset(tmp_path: Path) -> None:
    """With nothing configured, input paths stay None instead of guessing."""
    settings = load_settings(env={}, cwd=tmp_path)
    assert settings.config_file is None
    assert settings.paths.img_dir is None
    assert settings.paths.panel_dir is None
    assert settings.paths.boundary is None
    assert settings.paths.out_dir == DEFAULT_OUT_DIR
    assert settings.radiometric.saturation_dn == DEFAULT_SATURATION_DN
    assert settings.output.prefix == "field"


def test_missing_path_error_names_every_way_to_set_it(tmp_path: Path) -> None:
    settings = load_settings(env={}, cwd=tmp_path)
    with pytest.raises(ConfigError) as excinfo:
        settings.paths.require("img_dir", flag="--img-dir")
    message = str(excinfo.value)
    assert "--img-dir" in message
    assert "M3M_IMG_DIR" in message
    assert "m3m.example.toml" in message


def test_configured_but_absent_input_is_reported_clearly(tmp_path: Path) -> None:
    settings = load_settings(
        None, {"img_dir": str(tmp_path / "nope")}, env={}, cwd=tmp_path
    )
    with pytest.raises(ConfigError) as excinfo:
        settings.paths.require("img_dir", flag="--img-dir")
    message = str(excinfo.value)
    assert "does not exist" in message
    assert "nope" in message
    # The caller can still ask for the path without the existence check.
    assert settings.paths.require("img_dir", flag="--img-dir", must_exist=False).name == "nope"


def _write(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "m3m.toml"
    path.write_text(textwrap.dedent(body), encoding="utf-8")
    return path


def test_toml_relative_paths_resolve_against_the_config_directory(tmp_path: Path) -> None:
    nested = tmp_path / "configs"
    nested.mkdir()
    config_path = nested / "m3m.toml"
    config_path.write_text('[paths]\nimg_dir = "../captures"\n', encoding="utf-8")

    settings = load_settings(config_path, env={}, cwd=tmp_path)
    assert settings.paths.img_dir == (nested / "../captures").resolve()


def test_environment_beats_config_file(tmp_path: Path) -> None:
    config_path = _write(tmp_path, '[paths]\nimg_dir = "from_file"\n')
    settings = load_settings(
        config_path, env={"M3M_IMG_DIR": "from_env"}, cwd=tmp_path
    )
    assert settings.paths.img_dir == tmp_path / "from_env"
    assert settings.sources["paths.img_dir"] == "environment"


def test_command_line_beats_environment(tmp_path: Path) -> None:
    settings = load_settings(
        None,
        {"img_dir": "from_cli"},
        env={"M3M_IMG_DIR": "from_env"},
        cwd=tmp_path,
    )
    assert settings.paths.img_dir == tmp_path / "from_cli"
    assert settings.sources["paths.img_dir"] == "command line"


def test_none_overrides_do_not_mask_lower_priority_sources(tmp_path: Path) -> None:
    """A flag the user did not pass must not wipe an environment value."""
    settings = load_settings(
        None, {"img_dir": None}, env={"M3M_IMG_DIR": "from_env"}, cwd=tmp_path
    )
    assert settings.paths.img_dir == tmp_path / "from_env"


def test_config_file_is_discovered_in_a_parent_directory(tmp_path: Path) -> None:
    _write(tmp_path, '[paths]\nimg_dir = "captures"\n')
    child = tmp_path / "a" / "b"
    child.mkdir(parents=True)
    assert find_config_file(child, env={}) == (tmp_path / "m3m.toml").resolve()


def test_missing_explicit_config_file_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="Config file not found"):
        load_settings(tmp_path / "nope.toml", env={}, cwd=tmp_path)


def test_malformed_toml_reports_the_file(tmp_path: Path) -> None:
    path = tmp_path / "m3m.toml"
    path.write_text("[paths\nimg_dir = ", encoding="utf-8")
    with pytest.raises(ConfigError, match="not valid TOML"):
        load_settings(path, env={}, cwd=tmp_path)


@pytest.mark.parametrize(
    "roi",
    [
        "[10, 20, 5, 70]",  # x_min >= x_max
        "[10, 20, 50]",  # wrong arity
        "[10, 20, 50, 70, 90]",
        "[-1, 20, 50, 70]",  # negative origin
    ],
)
def test_invalid_panel_roi_is_rejected(tmp_path: Path, roi: str) -> None:
    config_path = _write(
        tmp_path, f'[radiometric.panel_roi]\nNIR = {roi}\n'
    )
    with pytest.raises(ConfigError):
        load_settings(config_path, env={}, cwd=tmp_path)


def test_panel_roi_from_cli_overrides_only_the_named_band(tmp_path: Path) -> None:
    config_path = _write(
        tmp_path,
        '[radiometric.panel_roi]\nG = [1, 2, 3, 4]\nNIR = [5, 6, 7, 8]\n',
    )
    settings = load_settings(
        config_path, {"panel_roi": {"NIR": (9, 10, 11, 12)}}, env={}, cwd=tmp_path
    )
    assert settings.radiometric.roi_for("NIR") == (9, 10, 11, 12)
    assert settings.radiometric.roi_for("G") == (1, 2, 3, 4)


def test_missing_panel_roi_explains_that_it_cannot_be_guessed(tmp_path: Path) -> None:
    settings = load_settings(env={}, cwd=tmp_path)
    with pytest.raises(ConfigError, match="capture-geometry specific"):
        settings.radiometric.roi_for("NIR")


def test_unknown_band_in_panel_reflectance_is_rejected(tmp_path: Path) -> None:
    config_path = _write(
        tmp_path, '[radiometric.panel_reflectance]\nBLUE = 0.5\n'
    )
    with pytest.raises(ConfigError, match="unknown bands"):
        load_settings(config_path, env={}, cwd=tmp_path)


@pytest.mark.parametrize("prefix", ["a/b", "a\\b", ""])
def test_output_prefix_must_be_a_plain_token(tmp_path: Path, prefix: str) -> None:
    with pytest.raises(ConfigError):
        load_settings(None, {"prefix": prefix}, env={}, cwd=tmp_path)


def test_settings_are_json_serialisable(tmp_path: Path) -> None:
    import json

    config_path = _write(
        tmp_path,
        '[paths]\nimg_dir = "c"\n[radiometric.panel_roi]\nNIR = [1, 2, 3, 4]\n',
    )
    payload = json.dumps(load_settings(config_path, env={}, cwd=tmp_path).as_dict())
    assert "img_dir" in payload
