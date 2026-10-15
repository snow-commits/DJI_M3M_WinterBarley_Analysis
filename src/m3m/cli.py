"""Command line interface: ``m3m <command>``.

Every subcommand is a thin wrapper around one library function. Configuration is
resolved once, from the command line, the ``M3M_*`` environment and ``m3m.toml``
(see :mod:`m3m.config`), so no subcommand contains a hard-coded path.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from m3m import __version__
from m3m.config import (
    ConfigError,
    Settings,
    load_settings,
    render_example_config,
)

PROGRAM = "m3m"

EXIT_OK = 0
EXIT_USAGE = 2

_OVERRIDE_PATH_KEYS = ("img_dir", "panel_dir", "boundary", "annotation_dir", "out_dir")


def build_parser() -> argparse.ArgumentParser:
    """Construct the full argument parser."""
    parser = argparse.ArgumentParser(
        prog=PROGRAM,
        description=(
            "Label-free processing toolkit for DJI Mavic 3 Multispectral imagery: "
            "radiometric calibration, vegetation indices, within-field relative "
            "zones and sensitivity analysis."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Configuration is read from --config, $M3M_CONFIG or ./m3m.toml, then "
            "from M3M_* environment variables, then from command line flags.\n"
            "Run `m3m config init` to write a starter m3m.toml, and "
            "`m3m synthetic --dest demo` to generate data you can run everything on."
        ),
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    # Also accepted before the subcommand, which is where most people type it.
    # Subcommand copies use SUPPRESS defaults so they never clobber this value.
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to m3m.toml (default: $M3M_CONFIG, else ./m3m.toml)",
    )
    parser.add_argument("--out-dir", type=Path, default=None, help="Override [paths] out_dir")
    subparsers = parser.add_subparsers(dest="command", metavar="<command>")

    _add_config_commands(subparsers)
    _add_audit(subparsers)
    _add_panel(subparsers)
    _add_calibrate(subparsers)
    _add_orthorectify(subparsers)
    _add_fill_nodata(subparsers)
    _add_assess(subparsers)
    _add_indices(subparsers)
    _add_zones(subparsers)
    _add_compare(subparsers)
    _add_sensitivity(subparsers)
    _add_prescribe(subparsers)
    _add_synthetic(subparsers)
    return parser


def _add_common(parser: argparse.ArgumentParser, *, out_dir: bool = True) -> None:
    # SUPPRESS keeps these keys out of the namespace when the flag is absent, so
    # a value given before the subcommand is not overwritten by a default.
    parser.add_argument(
        "--config",
        type=Path,
        default=argparse.SUPPRESS,
        help="Path to m3m.toml (default: $M3M_CONFIG, else ./m3m.toml)",
    )
    if out_dir:
        parser.add_argument(
            "--out-dir",
            type=Path,
            default=argparse.SUPPRESS,
            help="Override [paths] out_dir",
        )


def _add_input_paths(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--img-dir", type=Path, default=None, help="Raw M3M capture directory")
    parser.add_argument(
        "--panel-dir", type=Path, default=None, help="Reflectance panel capture directory"
    )
    parser.add_argument("--boundary", type=Path, default=None, help="Field boundary vector")


def _add_radiometric(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--saturation-dn", type=int, default=None, help="Digital number treated as saturated"
    )


def _add_config_commands(subparsers: Any) -> None:
    config = subparsers.add_parser("config", help="Inspect or create the configuration")
    config_sub = config.add_subparsers(dest="config_command", metavar="<action>")

    show = config_sub.add_parser("show", help="Print the resolved configuration as JSON")
    _add_common(show)
    show.set_defaults(handler=_handle_config_show)

    init = config_sub.add_parser("init", help="Write a starter m3m.toml")
    init.add_argument("--path", type=Path, default=Path("m3m.toml"), help="File to write")
    init.add_argument("--force", action="store_true", help="Overwrite an existing file")
    init.set_defaults(handler=_handle_config_init)


def _add_audit(subparsers: Any) -> None:
    parser = subparsers.add_parser(
        "audit", help="Inventory raw captures and report missing bands"
    )
    _add_common(parser)
    parser.add_argument("--img-dir", type=Path, default=None, help="Raw M3M capture directory")
    parser.add_argument(
        "--output", type=Path, default=None, help="Inventory CSV (default: <out>/dataset_inventory.csv)"
    )
    parser.set_defaults(handler=_handle_audit)


def _add_panel(subparsers: Any) -> None:
    parser = subparsers.add_parser(
        "panel", help="Inspect reflectance panel captures and derive the NIR anchor"
    )
    _add_common(parser)
    parser.add_argument("--panel-dir", type=Path, default=None)
    _add_radiometric(parser)
    parser.add_argument(
        "--show-roi",
        action="store_true",
        help="Write a PNG per band with the configured panel ROI outlined",
    )
    parser.add_argument(
        "--roi-band",
        action="append",
        default=None,
        metavar="BAND=XM0,YM0,XM1,YM1",
        help="Override [radiometric.panel_roi] for one band; repeatable",
    )
    parser.set_defaults(handler=_handle_panel)


def _add_calibrate(subparsers: Any) -> None:
    parser = subparsers.add_parser(
        "calibrate", help="Apply the DJI signal model and write reflectance GeoTIFFs"
    )
    _add_common(parser)
    _add_input_paths(parser)
    _add_radiometric(parser)
    parser.add_argument(
        "--limit", type=int, default=None, help="Process only the first N capture groups"
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Also write a one-image-per-band preview before the full run",
    )
    parser.set_defaults(handler=_handle_calibrate)


def _add_orthorectify(subparsers: Any) -> None:
    parser = subparsers.add_parser(
        "orthorectify", help="Render (and optionally run) the OpenDroneMap script"
    )
    _add_common(parser)
    _add_input_paths(parser)
    parser.add_argument(
        "--dataset-name", default=None, help="ODM dataset name (folder inside the project)"
    )
    parser.add_argument(
        "--orthophoto-cm", type=int, default=None, help="Orthophoto resolution in centimetres"
    )
    parser.add_argument(
        "--expected-frames",
        type=int,
        default=None,
        help="Fail the script unless exactly this many calibrated frames exist",
    )
    parser.add_argument("--odm-image", default="opendronemap/odm:latest")
    parser.add_argument("--wsl-distro", default="Ubuntu")
    parser.add_argument(
        "--run", action="store_true", help="Execute the rendered script through wsl.exe"
    )
    parser.set_defaults(handler=_handle_orthorectify)


def _add_fill_nodata(subparsers: Any) -> None:
    parser = subparsers.add_parser(
        "fill-nodata",
        help="Fill saturated NoData pixels in the ODM working copy (runs inside the container too)",
    )
    parser.add_argument("--images-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--audit-csv", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.set_defaults(handler=_handle_fill_nodata, config=None, out_dir=None)


def _add_assess(subparsers: Any) -> None:
    parser = subparsers.add_parser(
        "assess", help="Check a reflectance orthomosaic against physically valid ranges"
    )
    _add_common(parser)
    parser.add_argument("--raster", type=Path, required=True)
    parser.add_argument("--boundary", type=Path, default=None)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.set_defaults(handler=_handle_assess)


def _add_indices(subparsers: Any) -> None:
    parser = subparsers.add_parser(
        "indices", help="Compute NDVI / NDRE / EVI2 rasters and summary statistics"
    )
    _add_common(parser)
    parser.add_argument("--raster", type=Path, required=True, help="Reflectance orthomosaic")
    parser.add_argument("--boundary", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None, help="Default: <out>/indices")
    parser.add_argument("--prefix", default=None, help="Output filename prefix")
    parser.set_defaults(handler=_handle_indices)


def _add_zones(subparsers: Any) -> None:
    parser = subparsers.add_parser(
        "zones", help="Classify indices into within-field relative quintiles and map them"
    )
    _add_common(parser)
    parser.add_argument("--index-dir", type=Path, default=None, help="Default: <out>/indices")
    parser.add_argument("--boundary", type=Path, default=None)
    parser.add_argument("--prefix", default=None)
    parser.add_argument("--no-figures", action="store_true", help="Skip the PNG maps")
    parser.set_defaults(handler=_handle_zones)


def _add_compare(subparsers: Any) -> None:
    parser = subparsers.add_parser(
        "compare", help="Compare index pairs, or a repaired orthomosaic against the original"
    )
    _add_common(parser)
    parser.add_argument("--index-dir", type=Path, default=None, help="Default: <out>/indices")
    parser.add_argument("--prefix", default=None)
    parser.add_argument("--before", type=Path, default=None, help="Pre-repair orthomosaic")
    parser.add_argument("--after", type=Path, default=None, help="Repaired orthomosaic")
    parser.add_argument("--boundary", type=Path, default=None)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--output-figure", type=Path, default=None)
    parser.add_argument(
        "--patch-window",
        default=None,
        metavar="COL,ROW,WIDTH,HEIGHT",
        help="Before/after preview window in raster pixels (default: centred)",
    )
    parser.set_defaults(handler=_handle_compare)


def _add_sensitivity(subparsers: Any) -> None:
    parser = subparsers.add_parser(
        "sensitivity",
        help="Test how much relative zoning depends on index weights (not a prediction)",
    )
    _add_common(parser)
    parser.add_argument("--index-dir", type=Path, default=None, help="Default: <out>/indices")
    parser.add_argument("--boundary", type=Path, default=None)
    parser.add_argument("--prefix", default=None)
    parser.add_argument("--output-dir", type=Path, default=None, help="Default: <out>/sensitivity")
    parser.add_argument("--no-figures", action="store_true")
    parser.set_defaults(handler=_handle_sensitivity)


def _add_prescribe(subparsers: Any) -> None:
    parser = subparsers.add_parser(
        "prescribe",
        help="Turn relative zones into a variable-rate prescription with areas and costs",
        description=(
            "Builds a variable-rate application map from the within-field relative zones.\n"
            "The rate schedule is an operator decision, not a model output: this toolkit\n"
            "has no ground truth and cannot know how much product a zone needs."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _add_common(parser)
    parser.add_argument(
        "--class-raster",
        type=Path,
        default=None,
        help="Relative-class raster; default <out>/indices/<prefix>_<index>_relative_5class.tif",
    )
    parser.add_argument(
        "--index",
        default="ndvi",
        help="Which index's zones to use when --class-raster is not given (default: ndvi)",
    )
    parser.add_argument("--index-dir", type=Path, default=None, help="Default: <out>/indices")
    parser.add_argument("--boundary", type=Path, default=None, help="Only used to draw the figure")
    parser.add_argument("--output-dir", type=Path, default=None, help="Default: <out>/prescription")
    parser.add_argument(
        "--rate",
        action="append",
        default=None,
        metavar="CLASS=RATE",
        help=(
            "Override the rate schedule, e.g. --rate 1=0,2=40,3=80,4=120,5=160 "
            "(repeatable; later entries win)"
        ),
    )
    parser.add_argument(
        "--unit", default=None, help="Rate unit, e.g. kg/ha (default: from [prescription])"
    )
    parser.add_argument(
        "--target-mean-rate",
        type=float,
        default=None,
        help=(
            "Rescale the schedule so the area-weighted mean rate equals this value. Keeps "
            "the total product under operator control even though the zones are "
            "equal-frequency."
        ),
    )
    parser.add_argument("--product-price", type=float, default=None, help="Price per rate unit")
    parser.add_argument(
        "--application-cost-per-ha", type=float, default=None, help="Machine cost per hectare"
    )
    parser.add_argument("--currency", default=None, help="Currency label for the cost table")
    parser.add_argument("--no-figure", action="store_true", help="Skip the PNG map")
    parser.set_defaults(handler=_handle_prescribe)


def _add_synthetic(subparsers: Any) -> None:
    parser = subparsers.add_parser(
        "synthetic", help="Generate a small synthetic dataset so you can run everything offline"
    )
    _add_common(parser)
    parser.add_argument("--dest", type=Path, required=True, help="Directory to generate into")
    parser.add_argument("--groups", type=int, default=3, help="Number of capture groups")
    parser.add_argument("--seed", type=int, default=20260101)
    parser.add_argument(
        "--with-orthomosaic", action="store_true", default=True, help="Also write an orthomosaic"
    )
    parser.add_argument(
        "--no-orthomosaic", dest="with_orthomosaic", action="store_false"
    )
    parser.add_argument(
        "--config-path",
        type=Path,
        default=None,
        help="Write a matching m3m.toml here (default: <dest>/m3m.toml)",
    )
    parser.set_defaults(handler=_handle_synthetic)


# --------------------------------------------------------------------------- #
# handlers
# --------------------------------------------------------------------------- #


def _require_file(path: Path | None, flag: str) -> Path:
    """Fail with a readable message before the geospatial stack sees a bad path.

    ``pyogrio``/``rasterio`` raise driver-specific errors for a missing file
    that are neither ``FileNotFoundError`` nor ``OSError``, so checking here
    keeps the CLI's error handling honest.
    """
    if path is None:
        raise ConfigError(f"{flag} is required.")
    if not Path(path).is_file():
        raise ConfigError(f"{flag} does not exist: {path}")
    return Path(path)


def _resolve(args: argparse.Namespace) -> Settings:
    overrides: dict[str, Any] = {}
    for key in (*_OVERRIDE_PATH_KEYS, "prefix", "saturation_dn", "dataset_name",
                "orthophoto_resolution_cm"):
        value = getattr(args, key, None)
        if value is not None:
            overrides[key] = value
    roi_overrides = _parse_roi_overrides(getattr(args, "roi_band", None))
    if roi_overrides:
        overrides["panel_roi"] = roi_overrides
    prescription_overrides = _parse_prescription_overrides(args)
    if prescription_overrides:
        overrides["prescription"] = prescription_overrides
    return load_settings(getattr(args, "config", None), overrides)


def _parse_rate_table(values: Sequence[str] | None) -> dict[int, float]:
    """Parse ``--rate 1=0,2=40`` (and repeated ``--rate`` flags) into a mapping."""
    if not values:
        return {}
    rates: dict[int, float] = {}
    for chunk in values:
        for item in chunk.split(","):
            item = item.strip()
            if not item:
                continue
            if "=" not in item:
                raise ConfigError(f"--rate expects CLASS=RATE, got {item!r}")
            code, _, rate = item.partition("=")
            try:
                rates[int(code.strip())] = float(rate.strip())
            except ValueError as exc:
                raise ConfigError(f"--rate {item!r} is not CLASS=RATE with numbers") from exc
    return rates


def _parse_prescription_overrides(args: argparse.Namespace) -> dict[str, Any]:
    """Collect the prescription flags, ignoring the ones that were not given."""
    result: dict[str, Any] = {}
    rates = _parse_rate_table(getattr(args, "rate", None))
    if rates:
        result["rate_by_class"] = rates
    for attribute, setting in (
        ("unit", "unit"),
        ("target_mean_rate", "target_mean_rate"),
        ("product_price", "product_price_per_unit"),
        ("application_cost_per_ha", "application_cost_per_ha"),
        ("currency", "currency"),
    ):
        value = getattr(args, attribute, None)
        if value is not None:
            result[setting] = value
    return result


def _parse_roi_overrides(values: Sequence[str] | None) -> dict[str, tuple[int, int, int, int]]:
    if not values:
        return {}
    result: dict[str, tuple[int, int, int, int]] = {}
    for item in values:
        if "=" not in item:
            raise ConfigError(f"--roi-band expects BAND=x0,y0,x1,y1, got {item!r}")
        band, _, coordinates = item.partition("=")
        parts = [piece.strip() for piece in coordinates.split(",")]
        if len(parts) != 4:
            raise ConfigError(f"--roi-band {item!r} needs four integers")
        result[band.strip().upper()] = tuple(int(piece) for piece in parts)  # type: ignore[assignment]
    return result


def _handle_config_show(args: argparse.Namespace) -> int:
    settings = _resolve(args)
    print(json.dumps(settings.as_dict(), indent=2, ensure_ascii=False))
    return EXIT_OK


def _handle_config_init(args: argparse.Namespace) -> int:
    path: Path = args.path
    if path.exists() and not args.force:
        raise ConfigError(f"{path} already exists; pass --force to overwrite it.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_example_config(), encoding="utf-8")
    print(f"Wrote {path}\nEdit the [paths] table, then run `m3m config show` to confirm.")
    return EXIT_OK


def _handle_audit(args: argparse.Namespace) -> int:
    from m3m import audit

    settings = _resolve(args)
    img_dir = settings.paths.require("img_dir", flag="--img-dir")
    output = args.output or (settings.paths.out_dir / "dataset_inventory.csv")
    _, stats = audit.run(img_dir, output)
    print(f"Capture groups: {stats['groups']}")
    print(f"Complete groups: {stats['complete']}")
    print(f"Missing or abnormal groups: {stats['abnormal']}")
    print(f"Inventory: {output}")
    print(f"Elapsed: {stats['elapsed_seconds']:.2f} s")
    return EXIT_OK


def _handle_panel(args: argparse.Namespace) -> int:
    from m3m import radiometric
    from m3m.constants import BANDS
    from m3m.io import find_band

    settings = _resolve(args)
    panel_dir = settings.paths.require("panel_dir", flag="--panel-dir")
    radiometric_settings = settings.radiometric

    if radiometric_settings.panel_roi is None:
        raise ConfigError(
            "No panel ROI is configured. Panel ROIs depend on the capture geometry, "
            "so derive them from your own panel images first:\n"
            "  * measure the panel location in the raw captures, then\n"
            "  * set [radiometric.panel_roi] in m3m.toml, or pass\n"
            "    --roi-band NIR=x0,y0,x1,y1\n"
            "The synthetic dataset (`m3m synthetic --dest demo`) ships a working ROI "
            "you can compare against."
        )

    for band in BANDS:
        try:
            paths = find_band(panel_dir, band)
        except FileNotFoundError as exc:
            print(f"{band}: {exc}")
            continue
        if len(paths) > 1:
            print(f"{band}: {len(paths)} panel captures found; inspecting each")
        for path in paths:
            roi = radiometric_settings.roi_for(band)
            report = radiometric.inspect_panel_image(
                path, band, roi, radiometric_settings.saturation_dn
            )
            print(radiometric.describe_panel_report(report))
            if args.show_roi:
                target = settings.paths.out_dir / "panel_roi" / f"{path.stem}_roi.png"
                print(f"  ROI overlay: {radiometric.write_roi_overlay(path, roi, target)}")
        print()

    if not radiometric_settings.panel_roi:
        return EXIT_OK

    anchor = radiometric.calibrate_nir_anchor(
        panel_dir=panel_dir,
        panel_roi=radiometric_settings.panel_roi,
        panel_reflectance=radiometric_settings.panel_reflectance,
        saturation_dn=radiometric_settings.saturation_dn,
    )
    print("NIR panel anchor:")
    print(f"  nominal NIR reflectance : {anchor.panel_reflectance_nir:.4f}")
    print(f"  unsaturated ROI fraction: {100 * anchor.valid_fraction:.2f}%")
    print(f"  corrected panel signal  : {anchor.panel_signal:.8f}")
    print(f"  scene scale factor      : {anchor.scale:.8f}")
    print()
    print("Cross-band consistency against the panel certificate (diagnostic only):")
    for row in radiometric.check_cross_band_agreement(
        anchor,
        panel_dir=panel_dir,
        panel_roi=radiometric_settings.panel_roi,
        panel_reflectance=radiometric_settings.panel_reflectance,
        saturation_dn=radiometric_settings.saturation_dn,
    ):
        if row.get("saturated"):
            print(f"  {row['band']}: ROI fully saturated, no estimate")
        else:
            print(
                f"  {row['band']}: nominal={row['nominal']:.4f}, "
                f"predicted={row['predicted']:.4f}, "
                f"difference={row['relative_difference_percent']:+.2f}%"
            )
    return EXIT_OK


def _handle_calibrate(args: argparse.Namespace) -> int:
    from m3m import radiometric

    settings = _resolve(args)
    img_dir = settings.paths.require("img_dir", flag="--img-dir")
    panel_dir = settings.paths.require("panel_dir", flag="--panel-dir")
    radiometric_settings = settings.radiometric

    anchor = radiometric.calibrate_nir_anchor(
        panel_dir=panel_dir,
        panel_roi=radiometric_settings.panel_roi or {},
        panel_reflectance=radiometric_settings.panel_reflectance,
        saturation_dn=radiometric_settings.saturation_dn,
    )
    print(
        f"NIR panel anchor: nominal={anchor.panel_reflectance_nir:.4f}, "
        f"unsaturated={100 * anchor.valid_fraction:.2f}%, scale={anchor.scale:.8f}"
    )

    if args.preview:
        radiometric.preview_one_flight(
            anchor.scale,
            img_dir=img_dir,
            out_dir=settings.paths.out_dir,
            saturation_dn=radiometric_settings.saturation_dn,
            panel_reflectance_nir=anchor.panel_reflectance_nir,
        )

    summary = radiometric.process_dataset(
        anchor.scale,
        img_dir=img_dir,
        out_dir=settings.paths.out_dir,
        saturation_dn=radiometric_settings.saturation_dn,
        panel_reflectance_nir=anchor.panel_reflectance_nir,
        limit=args.limit,
    )
    if summary.failed:
        print(
            f"{summary.failed} image(s) failed; see {summary.manifest_path}",
            file=sys.stderr,
        )
        return 1
    return EXIT_OK


def _handle_orthorectify(args: argparse.Namespace) -> int:
    from m3m import orthorectification, radiometric

    settings = _resolve(args)
    img_dir = settings.paths.require("img_dir", flag="--img-dir")
    calib_dir = settings.paths.out_dir / "radiometric"
    manifest = calib_dir / "radiometric_manifest.csv"
    if not manifest.is_file():
        raise ConfigError(
            f"No calibration manifest at {manifest}. Run `m3m calibrate` first."
        )

    out_dir = settings.paths.out_dir
    project = out_dir / "odm_project"
    deliv = out_dir / "orthomosaic"
    script_path = out_dir / "orthorectify.sh"

    script = orthorectification.render_script(
        work_dir="/root/m3m_ortho",
        raw_src=img_dir,
        reflectance_src=calib_dir,
        project=project,
        deliver=deliv,
        manifest=manifest,
        dataset=settings.orthorectification.dataset_name,
        output_suffix=radiometric.OUTPUT_SUFFIX,
        orthophoto_resolution_cm=settings.orthorectification.orthophoto_resolution_cm,
        expected_frames=args.expected_frames,
        odm_image=args.odm_image,
    )
    orthorectification.write_script(script_path, script)
    print(f"Wrote {script_path}")
    print(
        "This script needs WSL with a Docker engine and the opendronemap/odm image. "
        "Review it before running it."
    )
    if args.run:
        print(f"Running through wsl.exe -d {args.wsl_distro} ...")
        orthorectification.run_via_wsl(script_path, distro=args.wsl_distro)
    return EXIT_OK


def _handle_fill_nodata(args: argparse.Namespace) -> int:
    from m3m import nodata

    nodata.run(
        _require_file(args.images_dir, "--images-dir"),
        _require_file(args.manifest, "--manifest"),
        args.audit_csv,
        dry_run=args.dry_run,
        limit=args.limit,
    )
    return EXIT_OK


def _handle_assess(args: argparse.Namespace) -> int:
    from m3m import quality

    settings = _resolve(args)
    boundary = args.boundary or settings.paths.require(
        "boundary", flag="--boundary"
    )
    output_json = args.output_json or (settings.paths.out_dir / "orthomosaic_quality.json")
    result = quality.run(_require_file(args.raster, "--raster"), boundary, output_json)
    print(quality.format_report(result))
    print(f"Wrote {output_json}")
    return EXIT_OK


def _default_index_dir(settings: Settings, override: Path | None) -> Path:
    return override or (settings.paths.out_dir / "indices")


def _handle_indices(args: argparse.Namespace) -> int:
    from m3m import indices

    settings = _resolve(args)
    boundary = args.boundary or settings.paths.require("boundary", flag="--boundary")
    output_dir = args.output_dir or (settings.paths.out_dir / "indices")
    rows = indices.calculate(
        _require_file(args.raster, "--raster"),
        boundary,
        output_dir,
        prefix=settings.output.prefix,
    )
    print(f"Wrote index rasters and summary to {output_dir}")
    print(indices.format_summary(rows))
    return EXIT_OK


def _handle_zones(args: argparse.Namespace) -> int:
    from m3m import zones

    settings = _resolve(args)
    boundary = args.boundary or settings.paths.require("boundary", flag="--boundary")
    index_dir = _default_index_dir(settings, args.index_dir)
    rows = zones.run(
        index_dir,
        boundary,
        prefix=settings.output.prefix,
        make_figures=not args.no_figures,
    )
    print(f"Wrote relative-class rasters, summary and maps to {index_dir}")
    for name in ("NDVI", "NDRE", "EVI2"):
        subset = [row for row in rows if row["index"] == name]
        if subset:
            thresholds = [row["upper_threshold"] for row in subset[:4]]
            print(f"{name} approximate 20/40/60/80% thresholds: {', '.join(thresholds)}")
    return EXIT_OK


def _handle_compare(args: argparse.Namespace) -> int:
    from m3m import compare

    settings = _resolve(args)
    if args.before and args.after:
        boundary = args.boundary or settings.paths.require("boundary", flag="--boundary")
        patch_window = None
        if args.patch_window:
            parts = [piece.strip() for piece in args.patch_window.split(",")]
            if len(parts) != 4:
                raise ConfigError("--patch-window expects COL,ROW,WIDTH,HEIGHT")
            patch_window = tuple(int(piece) for piece in parts)  # type: ignore[assignment]
        output_json = args.output_json or (
            settings.paths.out_dir / "orthomosaic" / "repair_comparison.json"
        )
        output_figure = args.output_figure or (
            settings.paths.out_dir / "orthomosaic" / "repair_before_after.png"
        )
        result = compare.compare_repairs(
            _require_file(args.before, "--before"),
            _require_file(args.after, "--after"),
            boundary,
            output_json=output_json,
            output_figure=output_figure,
            patch_window=patch_window,
        )
        print(f"Wrote {result.get('report')} and {result.get('figure')}")
        for name, stats in result["bands"].items():
            print(
                f"  {name}: common pixels={stats['common_valid_pixels']:,}, "
                f"changed>0.01={stats['changed_gt_0_01']:,}, "
                f"valid RMSE={stats['valid_rmse']}"
            )
        return EXIT_OK

    import csv

    index_dir = _default_index_dir(settings, args.index_dir)
    rows = compare.compare_indices(index_dir, prefix=settings.output.prefix)
    path = index_dir / "vegetation_index_pairwise_comparison.csv"
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {path}")
    for row in rows:
        print(
            f"{row['index_pair']}: Pearson r={row['pearson_r']:.4f}, "
            f"same class={row['same_quintile_class_percent']:.2f}%, "
            f"lowest/highest Jaccard={row['lowest_class_jaccard']:.4f}/"
            f"{row['highest_class_jaccard']:.4f}"
        )
    return EXIT_OK


def _handle_sensitivity(args: argparse.Namespace) -> int:
    from m3m import sensitivity

    settings = _resolve(args)
    boundary = args.boundary or settings.paths.require("boundary", flag="--boundary")
    index_dir = _default_index_dir(settings, args.index_dir)
    output_dir = args.output_dir or (settings.paths.out_dir / "sensitivity")
    result = sensitivity.run_experiment(
        index_dir,
        boundary,
        output_dir,
        prefix=settings.output.prefix,
        make_figures=not args.no_figures,
    )
    print(
        f"Wrote weight-sensitivity outputs to {output_dir}\n"
        f"Common valid pixels: {result['valid_pixels']:,}; "
        f"pixel area: {result['pixel_area_m2']:.6f} m2\n"
        "Reminder: this measures weight sensitivity, not predictive accuracy."
    )
    for row in result["pair_rows"]:
        print(
            f"  {row['candidate_a']} vs {row['candidate_b']}: "
            f"rho={row['spearman_rho_histogram_midranks']:.4f}, "
            f"same quintile={row['same_quintile_percent']:.2f}%"
        )
    return EXIT_OK


def _handle_prescribe(args: argparse.Namespace) -> int:
    from m3m import prescription
    from m3m.constants import INDEX_ORDER
    from m3m.zones import class_raster_path

    settings = _resolve(args)
    index_name = str(args.index).lower()
    if index_name not in INDEX_ORDER:
        raise ConfigError(
            f"--index must be one of {', '.join(INDEX_ORDER)}, got {args.index!r}"
        )

    index_dir = args.index_dir or (settings.paths.out_dir / "indices")
    class_raster = args.class_raster or class_raster_path(
        index_dir, settings.output.prefix, index_name
    )
    class_raster = _require_file(class_raster, "--class-raster")

    # The boundary is only needed to draw the figure, so its absence is not an error.
    boundary = args.boundary or settings.paths.boundary
    if boundary is not None and not Path(boundary).is_file():
        boundary = None

    rx = settings.prescription
    result = prescription.build(
        class_raster,
        output_dir=args.output_dir or (settings.paths.out_dir / "prescription"),
        prefix=settings.output.prefix,
        index_name=index_name,
        schedule=prescription.RateSchedule(rates=rx.rate_by_class, unit=rx.unit),
        target_mean_rate=rx.target_mean_rate,
        pricing=prescription.Pricing(
            product_price_per_unit=rx.product_price_per_unit,
            currency=rx.currency,
            application_cost_per_ha=rx.application_cost_per_ha,
        ),
        boundary=boundary,
        make_figure=not args.no_figure,
    )
    print(result.format_report())
    print()
    print(f"Rate raster : {result.rate_raster}")
    print(f"Summary CSV : {result.summary_csv}")
    print(f"Totals JSON : {result.totals_json}")
    if result.figure:
        print(f"Figure      : {result.figure}")
    return EXIT_OK


def _handle_synthetic(args: argparse.Namespace) -> int:
    from m3m import synthetic

    dataset = synthetic.generate_raw_dataset(
        args.dest, groups=args.groups, seed=args.seed
    )
    print(f"Raw captures : {dataset.img_dir} ({dataset.groups} groups x {len(synthetic.BANDS)} bands)")
    print(f"Panels       : {dataset.panel_dir}")

    orthomosaic = None
    if args.with_orthomosaic:
        orthomosaic = synthetic.generate_orthomosaic(args.dest, seed=args.seed)
        print(f"Orthomosaic  : {orthomosaic.raster}")
        print(f"Boundary     : {orthomosaic.boundary}")

    config_path = args.config_path or (Path(args.dest) / "m3m.toml")
    synthetic.write_config(config_path, dataset, orthomosaic=orthomosaic)
    print(f"Config       : {config_path}")
    print()
    print("Try the full pipeline:")
    print(f"  m3m audit      --config {config_path}")
    print(f"  m3m panel      --config {config_path}")
    print(f"  m3m calibrate  --config {config_path}")
    if orthomosaic is not None:
        print(
            f"  m3m indices    --config {config_path} "
            f"--raster {orthomosaic.raster}"
        )
    else:
        print(f"  m3m indices    --config {config_path} --raster <orthomosaic.tif>")
    print(f"  m3m zones      --config {config_path}")
    print(f"  m3m compare    --config {config_path}")
    print(f"  m3m sensitivity --config {config_path}")
    return EXIT_OK


# --------------------------------------------------------------------------- #
# entry point
# --------------------------------------------------------------------------- #


def _configure_runtime() -> None:
    """Bound GDAL's block cache unless the caller already chose a value.

    Measured on this toolkit's own workload, consuming a 12-megapixel
    orthomosaic (4000x3000, 48x the test-suite raster), by stage:

    ====================  ==============  =========  ===========
    stage                 GDAL default    64 MB cap  wall time
    ====================  ==============  =========  ===========
    ``zones.run``         +0.0 MB         +0.0 MB    --
    ``indices.calculate`` +444.9 MB       +115.2 MB  --
    peak working set      ~563 MB         ~234 MB    12.4 s
    ====================  ==============  =========  ===========

    Two conclusions. The accumulation and classification stages are exactly
    constant-memory, so the windowed reads are doing what the README claims. But
    GDAL's block cache, which defaults to about 5% of system RAM, dominated the
    measured peak and made it look as though memory scaled with image size.

    64 MB is the knee: peak drops from 563 MB to 234 MB with no measurable
    slowdown (12.4 s versus 12.5 s), while 32 MB costs 15 s. The windowed access
    pattern needs very little cache; only the overview build benefits from more.

    This lives in the CLI rather than the library so importing ``m3m`` never
    mutates process-wide GDAL state behind the caller's back. Set
    ``GDAL_CACHEMAX`` yourself to override it.
    """
    os.environ.setdefault("GDAL_CACHEMAX", "64")


def main(argv: Sequence[str] | None = None) -> int:
    """Parse arguments, dispatch, and turn known failures into clean messages."""
    _configure_runtime()
    parser = build_parser()
    args = parser.parse_args(argv)
    handler = getattr(args, "handler", None)
    if handler is None:
        parser.print_help()
        return EXIT_USAGE

    try:
        return int(handler(args))
    except (ConfigError, OSError, NotADirectoryError, ValueError) as exc:
        print(f"{PROGRAM}: error: {exc}", file=sys.stderr)
        return EXIT_USAGE
    except KeyboardInterrupt:  # pragma: no cover - interactive only
        print(f"{PROGRAM}: interrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
