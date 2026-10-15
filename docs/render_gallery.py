"""Render the README gallery figure from the synthetic demo dataset.

Deterministic: the synthetic generator is seeded, so re-running this reproduces
the same bytes. Writes docs/demo-pipeline.png inside the repository.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import rasterio  # noqa: E402
from matplotlib.colors import BoundaryNorm, ListedColormap  # noqa: E402
from rasterio.enums import Resampling  # noqa: E402

from m3m import indices, prescription, synthetic, zones  # noqa: E402

SHAPE = (600, 700)
DOWN = 4
ZONE_COLORS = ["#a50026", "#f46d43", "#fee08b", "#a6d96a", "#1a9850"]


def read_downsampled(path: Path, band: int = 1):
    with rasterio.open(path) as src:
        out_shape = (max(1, src.height // DOWN), max(1, src.width // DOWN))
        array = src.read(
            band, out_shape=out_shape, masked=True, resampling=Resampling.average
        )
        bounds = src.bounds
        crs = src.crs
    extent = (bounds.left, bounds.right, bounds.bottom, bounds.top)
    return array, extent, crs


def stretch(band: np.ndarray, low: float = 2.0, high: float = 98.0) -> np.ndarray:
    """Per-channel percentile stretch so the composite actually shows structure."""
    lo, hi = np.percentile(band, (low, high))
    if hi <= lo:
        return np.zeros_like(band)
    return np.clip((band - lo) / (hi - lo), 0, 1)


def main(argv: list[str]) -> int:
    repo = Path(argv[1]).resolve()
    work = Path(argv[2]).resolve()
    work.mkdir(parents=True, exist_ok=True)

    dataset = synthetic.generate_raw_dataset(
        work / "raw", groups=1, shape=(96, 128), panel_roi=(10, 20, 50, 70), with_rgb=False
    )
    ortho = synthetic.generate_orthomosaic(work, shape=SHAPE)
    config = synthetic.write_config(work / "m3m.toml", dataset, orthomosaic=ortho)

    index_dir = work / "outputs" / "indices"
    indices.calculate(ortho.raster, ortho.boundary, index_dir, prefix="demo")
    zones.run(index_dir, ortho.boundary, prefix="demo", make_figures=False)
    result = prescription.build(
        index_dir / "demo_ndvi_relative_5class.tif",
        output_dir=work / "outputs" / "prescription",
        prefix="demo",
        schedule=prescription.RateSchedule(
            {1: 0.0, 2: 60.0, 3: 120.0, 4: 180.0, 5: 240.0}, unit="kg N/ha"
        ),
        boundary=ortho.boundary,
        make_figure=False,
    )
    print("config:", config)

    import geopandas as gpd

    with rasterio.open(ortho.raster) as src:
        out_shape = (max(1, src.height // DOWN), max(1, src.width // DOWN))
        red = src.read(1, out_shape=out_shape, resampling=Resampling.average)
        green = src.read(2, out_shape=out_shape, resampling=Resampling.average)
        nir = src.read(3, out_shape=out_shape, resampling=Resampling.average)
        extent = (
            src.bounds.left,
            src.bounds.right,
            src.bounds.bottom,
            src.bounds.top,
        )
        crs = src.crs
    # NIR in the red channel, Red in green, Green in blue, each stretched on its
    # own percentiles -- a fixed divisor washes the whole scene out to flat red.
    composite = np.dstack([stretch(nir), stretch(red), stretch(green)])

    ndvi, _, _ = read_downsampled(index_dir / "demo_ndvi.tif")
    classes, _, _ = read_downsampled(index_dir / "demo_ndvi_relative_5class.tif")
    rate, _, _ = read_downsampled(result.rate_raster)

    field = gpd.read_file(ortho.boundary).to_crs(crs)

    figure, axes = plt.subplots(1, 4, figsize=(16.5, 4.9), constrained_layout=True)

    axes[0].imshow(composite, extent=extent, origin="upper")
    axes[0].set_title("Reflectance orthomosaic\nNIR–Red–Green false colour", fontsize=10)

    values = ndvi.compressed()
    axes[1].imshow(
        ndvi,
        extent=extent,
        origin="upper",
        cmap="RdYlGn",
        vmin=float(np.percentile(values, 2)),
        vmax=float(np.percentile(values, 98)),
    )
    axes[1].set_title("NDVI\n(NIR − Red) / (NIR + Red)", fontsize=10)

    zone_cmap = ListedColormap(ZONE_COLORS)
    axes[2].imshow(
        classes,
        extent=extent,
        origin="upper",
        cmap=zone_cmap,
        norm=BoundaryNorm(np.arange(0.5, 6.5, 1), 5),
        interpolation="nearest",
    )
    axes[2].set_title("Relative zones\nequal-frequency quintiles", fontsize=10)

    distinct = sorted({float(row["rate"]) for row in result.rows})
    step = min(b - a for a, b in zip(distinct, distinct[1:], strict=False)) / 2 or 1.0
    axes[3].imshow(
        rate,
        extent=extent,
        origin="upper",
        cmap=ListedColormap(ZONE_COLORS[: len(distinct)]),
        norm=BoundaryNorm([distinct[0] - step, *[v + step for v in distinct]], len(distinct)),
        interpolation="nearest",
    )
    axes[3].set_title("Prescription\n0–240 kg N/ha by zone", fontsize=10)

    for ax in axes:
        field.boundary.plot(ax=ax, color="#101010", linewidth=0.9)
        ax.set_xlabel("Easting (m)", fontsize=7.5)
        ax.tick_params(labelsize=6.5)
        ax.set_aspect("equal")
    axes[0].set_ylabel("Northing (m)", fontsize=7.5)

    figure.suptitle(
        "m3m pipeline on the synthetic demo dataset "
        "(`m3m synthetic`, 600x700 px). Reflectance is provisional and the rate "
        "schedule is an operator input, not a validated recommendation.",
        fontsize=9,
    )

    out = repo / "docs" / "demo-pipeline.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(figure)
    print("wrote", out, out.stat().st_size, "bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
