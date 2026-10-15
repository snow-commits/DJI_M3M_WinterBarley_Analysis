"""Configuration resolution for the M3M toolkit.

Precedence, lowest to highest:

1. built-in defaults
2. a TOML configuration file (``--config``, ``$M3M_CONFIG`` or ``./m3m.toml``)
3. ``M3M_*`` environment variables
4. explicit command-line overrides

No machine-specific path is baked into the source. Every input path must be
supplied through one of those sources; when it is missing the raised
:class:`ConfigError` names the flag, the environment variable and the TOML key
that would fix it.

Relative paths written in a TOML file are resolved against that file's own
directory, so a shared config behaves the same regardless of the current
working directory. Paths from the command line or the environment are resolved
against the current working directory, which is what a shell user expects.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:  # Python 3.11+
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - exercised only on Python 3.10
    import tomli as tomllib  # type: ignore[no-redef]

from m3m.constants import BANDS

ENV_PREFIX = "M3M_"

#: File names searched in the working directory when no config is given.
DEFAULT_CONFIG_NAMES: tuple[str, ...] = ("m3m.toml", "m3m.local.toml")

DEFAULT_OUT_DIR = Path("outputs")

#: Digital number at or above which a pixel is treated as saturated.
DEFAULT_SATURATION_DN = 65520

#: Fallback panel reflectance values (nominal, from a calibration certificate).
#: These only apply when neither the TOML file nor the CLI provides values.
DEFAULT_PANEL_REFLECTANCE: dict[str, float] = {
    "G": 0.493,
    "R": 0.494,
    "RE": 0.494,
    "NIR": 0.493,
}

DEFAULT_DATASET_NAME = "field"
DEFAULT_ORTHOPHOTO_RESOLUTION_CM = 5

#: Filename prefix for generated rasters, e.g. ``field_ndvi.tif``. Replaces the
#: dataset-specific literal that used to be baked into the analysis scripts.
DEFAULT_OUTPUT_PREFIX = "field"

_PATH_FIELDS: tuple[str, ...] = (
    "img_dir",
    "panel_dir",
    "boundary",
    "annotation_dir",
    "out_dir",
)

_FIELD_TO_ENV: dict[str, str] = {
    "img_dir": "M3M_IMG_DIR",
    "panel_dir": "M3M_PANEL_DIR",
    "boundary": "M3M_BOUNDARY",
    "annotation_dir": "M3M_ANNOTATION_DIR",
    "out_dir": "M3M_OUT_DIR",
}

#: How each path setting is described in error messages.
_PATH_PURPOSE: dict[str, str] = {
    "img_dir": "The raw M3M capture directory",
    "panel_dir": "The reflectance panel capture directory",
    "boundary": "The field boundary vector",
    "annotation_dir": "The annotation directory",
    "out_dir": "The output directory",
}


class ConfigError(ValueError):
    """Raised when configuration is missing, unreadable or inconsistent."""


def _missing_path_message(field_name: str, flag: str) -> str:
    env_name = _FIELD_TO_ENV.get(field_name, f"{ENV_PREFIX}{field_name.upper()}")
    purpose = _PATH_PURPOSE.get(field_name, field_name)
    return (
        f"{purpose} is not configured.\n"
        f"  Set it in one of these ways:\n"
        f"    * command line   {flag} <PATH>\n"
        f"    * environment    {env_name}=<PATH>\n"
        f"    * config file    [paths] {field_name} = \"<PATH>\"\n"
        f"  See m3m.example.toml for a ready-to-copy template."
    )


@dataclass
class Paths:
    """Filesystem inputs and the output root."""

    img_dir: Path | None = None
    panel_dir: Path | None = None
    boundary: Path | None = None
    annotation_dir: Path | None = None
    out_dir: Path = DEFAULT_OUT_DIR

    def require(self, field_name: str, *, flag: str, must_exist: bool = True) -> Path:
        """Return a configured path, or raise a message explaining how to set it.

        Existence is checked here rather than deeper in the stack because the
        geospatial libraries raise heterogeneous, hard-to-read errors for a
        missing file (``pyogrio`` raises neither ``FileNotFoundError`` nor
        ``OSError``), which would otherwise surface as a raw traceback.
        """
        value = getattr(self, field_name, None)
        if value is None:
            raise ConfigError(_missing_path_message(field_name, flag))
        path = Path(value)
        if must_exist and not path.exists():
            purpose = _PATH_PURPOSE.get(field_name, field_name)
            env_name = _FIELD_TO_ENV.get(field_name, "<none>")
            raise ConfigError(
                f"{purpose} does not exist: {path}\n"
                f"  Currently supplied by {flag} / {env_name} / [paths] {field_name}. "
                f"Run `m3m config show` to see which source won."
            )
        return path


@dataclass
class RadiometricSettings:
    """Parameters of the DJI per-image signal model and panel anchoring."""

    saturation_dn: int = DEFAULT_SATURATION_DN
    panel_reflectance: dict[str, float] = field(
        default_factory=lambda: dict(DEFAULT_PANEL_REFLECTANCE)
    )
    panel_roi: dict[str, tuple[int, int, int, int]] | None = None

    def reflectance_for(self, band: str) -> float:
        try:
            return float(self.panel_reflectance[band])
        except KeyError as exc:
            raise ConfigError(
                f"No panel reflectance configured for band {band!r}. "
                f"Known bands: {', '.join(sorted(self.panel_reflectance)) or 'none'}. "
                f"Set [radiometric.panel_reflectance] in m3m.toml."
            ) from exc

    def roi_for(self, band: str) -> tuple[int, int, int, int]:
        """Panel ROI as ``(x_min, y_min, x_max, y_max)`` in panel-image pixels."""
        roi = (self.panel_roi or {}).get(band)
        if roi is None:
            raise ConfigError(
                f"No panel ROI configured for band {band!r}.\n"
                f"  Panel ROIs are capture-geometry specific and cannot be guessed.\n"
                f"  Derive them from your own panel images and set them under\n"
                f"  [radiometric.panel_roi] in m3m.toml, for example:\n"
                f"      {band} = [x_min, y_min, x_max, y_max]"
            )
        if len(roi) != 4:
            raise ConfigError(f"Panel ROI for {band!r} must have four values, got {roi!r}.")
        x_min, y_min, x_max, y_max = (int(value) for value in roi)
        if not (0 <= x_min < x_max and 0 <= y_min < y_max):
            raise ConfigError(
                f"Panel ROI for {band!r} must satisfy 0 <= x_min < x_max and "
                f"0 <= y_min < y_max, got {roi!r}."
            )
        return x_min, y_min, x_max, y_max


@dataclass
class OrthorectificationSettings:
    """Knobs for the OpenDroneMap orchestration step."""

    dataset_name: str = DEFAULT_DATASET_NAME
    orthophoto_resolution_cm: int = DEFAULT_ORTHOPHOTO_RESOLUTION_CM


@dataclass
class OutputSettings:
    """Naming of generated artefacts."""

    prefix: str = DEFAULT_OUTPUT_PREFIX


@dataclass
class Settings:
    """Fully resolved configuration."""

    paths: Paths = field(default_factory=Paths)
    radiometric: RadiometricSettings = field(default_factory=RadiometricSettings)
    orthorectification: OrthorectificationSettings = field(
        default_factory=OrthorectificationSettings
    )
    output: OutputSettings = field(default_factory=OutputSettings)
    config_file: Path | None = None
    #: Which source supplied each key, for ``m3m config show``.
    sources: dict[str, str] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view of the resolved settings."""
        return {
            "config_file": str(self.config_file) if self.config_file else None,
            "paths": {
                name: (str(getattr(self.paths, name)) if getattr(self.paths, name) else None)
                for name in _PATH_FIELDS
            },
            "radiometric": {
                "saturation_dn": self.radiometric.saturation_dn,
                "panel_reflectance": dict(self.radiometric.panel_reflectance),
                "panel_roi": (
                    {band: list(roi) for band, roi in self.radiometric.panel_roi.items()}
                    if self.radiometric.panel_roi
                    else None
                ),
            },
            "orthorectification": {
                "dataset_name": self.orthorectification.dataset_name,
                "orthophoto_resolution_cm": self.orthorectification.orthophoto_resolution_cm,
            },
            "output": {"prefix": self.output.prefix},
            "sources": dict(self.sources),
        }


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #


def find_config_file(start: Path | None = None, env: Mapping[str, str] | None = None) -> Path | None:
    """Return the config file to use, or ``None`` when there is none.

    ``$M3M_CONFIG`` wins; otherwise the default names are looked up in *start*
    (the working directory by default) and then in each of its parents, so that
    running ``m3m`` from a subdirectory of a configured project still works.
    """
    env = os.environ if env is None else env
    explicit = env.get(f"{ENV_PREFIX}CONFIG")
    if explicit:
        path = Path(explicit).expanduser()
        if not path.is_file():
            raise ConfigError(f"{ENV_PREFIX}CONFIG points at a missing file: {path}")
        return path.resolve()

    directory = Path.cwd() if start is None else Path(start)
    for candidate_dir in (directory, *directory.parents):
        for name in DEFAULT_CONFIG_NAMES:
            candidate = candidate_dir / name
            if candidate.is_file():
                return candidate.resolve()
    return None


def read_config_file(path: Path) -> dict[str, Any]:
    """Parse a TOML config file, raising :class:`ConfigError` on malformed input."""
    try:
        with Path(path).open("rb") as stream:
            return tomllib.load(stream)
    except FileNotFoundError as exc:
        raise ConfigError(f"Config file not found: {path}") from exc
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"Config file is not valid TOML: {path}\n  {exc}") from exc


def _as_coordinate_list(value: Any, *, band: str) -> tuple[int, int, int, int]:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ConfigError(
            f"[radiometric.panel_roi] {band} must be a list of four integers "
            f"[x_min, y_min, x_max, y_max], got {value!r}."
        )
    try:
        x_min, y_min, x_max, y_max = (int(item) for item in value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(
            f"[radiometric.panel_roi] {band} must contain integers, got {value!r}."
        ) from exc
    if not (0 <= x_min < x_max and 0 <= y_min < y_max):
        raise ConfigError(
            f"[radiometric.panel_roi] {band} must satisfy 0 <= x_min < x_max and "
            f"0 <= y_min < y_max, got {value!r}."
        )
    return x_min, y_min, x_max, y_max


def _coerce_path(value: Any, *, base: Path | None) -> Path:
    if isinstance(value, Path):
        path = value
    elif isinstance(value, str):
        path = Path(os.path.expandvars(value)).expanduser()
    else:
        raise ConfigError(f"Expected a path string, got {type(value).__name__}: {value!r}")
    if base is not None:
        # Resolve against the config file's directory and normalise, so a value
        # like "../captures" surfaces as a clean absolute path in `config show`.
        path = (path if path.is_absolute() else base / path).resolve()
    return path


def load_settings(
    config_file: Path | str | None = None,
    overrides: Mapping[str, Any] | None = None,
    *,
    env: Mapping[str, str] | None = None,
    cwd: Path | None = None,
) -> Settings:
    """Resolve the effective settings.

    Parameters
    ----------
    config_file:
        Explicit config file. When omitted, :func:`find_config_file` decides.
    overrides:
        Highest-priority values, typically straight from parsed CLI arguments.
        Keys use the settings field names (``img_dir``, ``saturation_dn``,
        ``panel_roi``, ``dataset_name`` ...). ``None`` values are ignored so a
        flag that was not given never masks a lower-priority source.
    env:
        Environment mapping; defaults to :data:`os.environ`.
    cwd:
        Directory used to resolve relative command-line/environment paths.
    """
    env = os.environ if env is None else env
    overrides = {key: value for key, value in (overrides or {}).items() if value is not None}
    base_dir = Path.cwd() if cwd is None else Path(cwd)

    if config_file is None:
        resolved_config = find_config_file(base_dir, env)
    else:
        resolved_config = Path(config_file).expanduser().resolve()
        if not resolved_config.is_file():
            raise ConfigError(f"Config file not found: {resolved_config}")

    raw: dict[str, Any] = {}
    config_dir: Path | None = None
    if resolved_config is not None:
        raw = read_config_file(resolved_config)
        config_dir = resolved_config.parent

    settings = Settings(config_file=resolved_config)

    # ---- paths ---------------------------------------------------------- #
    raw_paths = raw.get("paths", {})
    if not isinstance(raw_paths, dict):
        raise ConfigError("[paths] must be a TOML table.")
    for name in _PATH_FIELDS:
        source = "default"
        value: Path | None = (
            DEFAULT_OUT_DIR if name == "out_dir" else None
        )
        if name in raw_paths:
            value = _coerce_path(raw_paths[name], base=config_dir)
            source = "config file"
        env_name = _FIELD_TO_ENV.get(name)
        if env_name and env.get(env_name):
            value = _coerce_path(env[env_name], base=None)
            if not value.is_absolute():
                value = base_dir / value
            source = "environment"
        if name in overrides:
            value = _coerce_path(overrides[name], base=None)
            if not value.is_absolute():
                value = base_dir / value
            source = "command line"
        setattr(settings.paths, name, value)
        settings.sources[f"paths.{name}"] = source

    # ---- radiometric ---------------------------------------------------- #
    raw_radio = raw.get("radiometric", {})
    if not isinstance(raw_radio, dict):
        raise ConfigError("[radiometric] must be a TOML table.")

    saturation_source = "default"
    saturation_dn = DEFAULT_SATURATION_DN
    if "saturation_dn" in raw_radio:
        saturation_dn = int(raw_radio["saturation_dn"])
        saturation_source = "config file"
    if env.get(f"{ENV_PREFIX}SATURATION_DN"):
        saturation_dn = int(env[f"{ENV_PREFIX}SATURATION_DN"])
        saturation_source = "environment"
    if "saturation_dn" in overrides:
        saturation_dn = int(overrides["saturation_dn"])
        saturation_source = "command line"
    settings.radiometric.saturation_dn = saturation_dn
    settings.sources["radiometric.saturation_dn"] = saturation_source

    reflectance = dict(DEFAULT_PANEL_REFLECTANCE)
    reflectance_source = "default"
    if isinstance(raw_radio.get("panel_reflectance"), dict):
        reflectance.update(
            {str(band).upper(): float(value) for band, value in raw_radio["panel_reflectance"].items()}
        )
        reflectance_source = "config file"
    if "panel_reflectance" in overrides:
        reflectance.update(
            {str(band).upper(): float(value) for band, value in overrides["panel_reflectance"].items()}
        )
        reflectance_source = "command line"
    settings.radiometric.panel_reflectance = reflectance
    settings.sources["radiometric.panel_reflectance"] = reflectance_source

    panel_roi: dict[str, tuple[int, int, int, int]] | None = None
    roi_source = "unset"
    if isinstance(raw_radio.get("panel_roi"), dict):
        panel_roi = {
            str(band).upper(): _as_coordinate_list(value, band=str(band).upper())
            for band, value in raw_radio["panel_roi"].items()
        }
        roi_source = "config file"
    if "panel_roi" in overrides:
        overrides_roi = overrides["panel_roi"]
        if isinstance(overrides_roi, dict):
            panel_roi = dict(panel_roi or {})
            panel_roi.update(
                {
                    str(band).upper(): _as_coordinate_list(value, band=str(band).upper())
                    for band, value in overrides_roi.items()
                }
            )
            roi_source = "command line"
    settings.radiometric.panel_roi = panel_roi
    settings.sources["radiometric.panel_roi"] = roi_source

    # ---- orthorectification --------------------------------------------- #
    raw_ortho = raw.get("orthorectification", {})
    if not isinstance(raw_ortho, dict):
        raise ConfigError("[orthorectification] must be a TOML table.")
    dataset_name = str(raw_ortho.get("dataset_name", DEFAULT_DATASET_NAME))
    resolution = int(
        raw_ortho.get("orthophoto_resolution_cm", DEFAULT_ORTHOPHOTO_RESOLUTION_CM)
    )
    if "dataset_name" in overrides:
        dataset_name = str(overrides["dataset_name"])
    if "orthophoto_resolution_cm" in overrides:
        resolution = int(overrides["orthophoto_resolution_cm"])
    settings.orthorectification.dataset_name = dataset_name
    settings.orthorectification.orthophoto_resolution_cm = resolution

    # ---- output naming --------------------------------------------------- #
    raw_output = raw.get("output", {})
    if not isinstance(raw_output, dict):
        raise ConfigError("[output] must be a TOML table.")
    prefix = str(raw_output.get("prefix", DEFAULT_OUTPUT_PREFIX))
    prefix_source = "config file" if "prefix" in raw_output else "default"
    if env.get(f"{ENV_PREFIX}OUTPUT_PREFIX"):
        prefix = env[f"{ENV_PREFIX}OUTPUT_PREFIX"]
        prefix_source = "environment"
    if "prefix" in overrides:
        prefix = str(overrides["prefix"])
        prefix_source = "command line"
    if not prefix or "/" in prefix or "\\" in prefix:
        raise ConfigError(f"Output prefix must be a plain filename token, got {prefix!r}.")
    settings.output.prefix = prefix
    settings.sources["output.prefix"] = prefix_source

    _validate_bands(settings)
    return settings


def _validate_bands(settings: Settings) -> None:
    """Reject panel reflectance entries that do not name a known band."""
    unknown = set(settings.radiometric.panel_reflectance) - set(BANDS)
    if unknown:
        raise ConfigError(
            f"[radiometric.panel_reflectance] names unknown bands: {', '.join(sorted(unknown))}. "
            f"Valid bands: {', '.join(BANDS)}."
        )


#: Used by ``m3m config init`` when the shipped example file is not reachable
#: (for example inside a wheel install), so the command always yields a usable
#: starting point.
_FALLBACK_TEMPLATE = """\
# m3m configuration. See the project README for the full schema.

[paths]
img_dir = "data/Dataset"
panel_dir = "data/Reflectance_Panel"
boundary = "data/boundary.gpkg"
out_dir = "outputs"

[radiometric]
saturation_dn = 65520

[radiometric.panel_reflectance]
G = 0.493
R = 0.494
RE = 0.494
NIR = 0.493

# Panel ROIs are specific to one capture geometry: re-derive them for your own
# panel images before running `m3m calibrate`.
[radiometric.panel_roi]
G = [0, 0, 1, 1]
R = [0, 0, 1, 1]
RE = [0, 0, 1, 1]
NIR = [0, 0, 1, 1]

[output]
prefix = "field"
"""


def render_example_config() -> str:
    """Return the documented example configuration."""
    template = Path(__file__).resolve().parents[2] / "m3m.example.toml"
    if template.is_file():
        return template.read_text(encoding="utf-8")
    return _FALLBACK_TEMPLATE
