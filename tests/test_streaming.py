"""Streaming correctness across many raster windows.

Every other test in the suite uses a 96x128 raster, which fits inside a single
512-pixel block. The windowed loops -- the code path the README's memory claim
depends on -- therefore never ran more than once. These tests force several
windows and check the result against a plain in-memory computation, because a
streaming accumulation bug (a histogram added twice, a mask applied to the wrong
window) is invisible on a single block.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.features import geometry_mask
from shapely.geometry import mapping

from m3m import indices, synthetic, zones

#: 600x700 with 512-pixel index blocks gives a 2x2 window grid.
MULTI_SHAPE = (600, 700)
EXPECTED_WINDOWS = 4


@pytest.fixture(scope="module")
def multi_block_ortho(tmp_path_factory: pytest.TempPathFactory):
    root = tmp_path_factory.mktemp("multiblock")
    return synthetic.generate_orthomosaic(root, shape=MULTI_SHAPE)


def _valid_mask(raster: Path, boundary: Path) -> np.ndarray:
    """Recompute the tool's valid mask independently, over the whole raster."""
    import geopandas as gpd

    with rasterio.open(raster) as src:
        red = src.read(1)
        nir = src.read(3)
        rededge = src.read(4)
        alpha = src.read(5)
        shapes = [mapping(g) for g in gpd.read_file(boundary).to_crs(src.crs).geometry]
        inside = geometry_mask(
            shapes,
            out_shape=(src.height, src.width),
            transform=src.transform,
            invert=True,
        )
    return (
        inside
        & (alpha > 0)
        & (red >= 0)
        & (red <= 1)
        & (nir >= 0)
        & (nir <= 1)
        & (rededge >= 0)
        & (rededge <= 1)
    )


def test_the_fixture_actually_spans_multiple_windows(multi_block_ortho) -> None:
    """Guard the guard: if this ever fits in one block, the tests below prove nothing."""
    with rasterio.open(multi_block_ortho.raster) as src:
        windows = list(src.block_windows(1))
    assert len(windows) > 1


def test_index_statistics_accumulate_correctly_across_windows(
    tmp_path: Path, multi_block_ortho
) -> None:
    output_dir = tmp_path / "indices"
    rows = indices.calculate(
        multi_block_ortho.raster, multi_block_ortho.boundary, output_dir, prefix="mb"
    )

    mask = _valid_mask(multi_block_ortho.raster, multi_block_ortho.boundary)
    expected_pixels = int(mask.sum())
    assert expected_pixels > 0

    # The streamed pixel count must equal the whole-raster count, not the count
    # of one window times four.
    for row in rows:
        assert row["valid_pixels"] == expected_pixels

    with rasterio.open(multi_block_ortho.raster) as src:
        red = src.read(1)
        nir = src.read(3)
        rededge = src.read(4)

    expected = {
        "NDVI": (nir - red) / (nir + red),
        "NDRE": (nir - rededge) / (nir + rededge),
        "EVI2": 2.5 * (nir - red) / (nir + 2.4 * red + 1),
    }
    for name, values in expected.items():
        with rasterio.open(output_dir / f"mb_{name.lower()}.tif") as src:
            written = src.read(1)
        assert np.allclose(written[mask], values[mask], atol=1e-5), name
        assert np.all(written[~mask] == indices.NODATA)


def test_histogram_quantiles_do_not_double_count_windows(
    tmp_path: Path, multi_block_ortho
) -> None:
    """A histogram folded in twice would shift the median away from the true one."""
    output_dir = tmp_path / "indices"
    rows = indices.calculate(
        multi_block_ortho.raster, multi_block_ortho.boundary, output_dir, prefix="mb"
    )
    mask = _valid_mask(multi_block_ortho.raster, multi_block_ortho.boundary)
    with rasterio.open(output_dir / "mb_ndvi.tif") as src:
        ndvi = src.read(1)

    median = float(np.median(ndvi[mask]))
    row = next(row for row in rows if row["index"] == "NDVI")
    # 100k bins over [-1, 1] is a 2e-5 bin width; the estimate is a midpoint.
    assert row["median_histogram_estimate"] == pytest.approx(median, abs=2e-4)
    assert row["mean"] == pytest.approx(float(ndvi[mask].mean()), rel=1e-6)
    assert row["p95_histogram_estimate"] == pytest.approx(
        float(np.percentile(ndvi[mask], 95)), abs=2e-4
    )


def test_zoning_stays_equal_frequency_across_windows(
    tmp_path: Path, multi_block_ortho
) -> None:
    """Classification counts must be accumulated per window, not per raster."""
    index_dir = tmp_path / "indices"
    indices.calculate(
        multi_block_ortho.raster, multi_block_ortho.boundary, index_dir, prefix="mb"
    )
    rows = zones.run(
        index_dir, multi_block_ortho.boundary, prefix="mb", make_figures=False
    )

    assert len(rows) == 15
    for name in ("NDVI", "NDRE", "EVI2"):
        subset = [row for row in rows if row["index"] == name]
        assert sum(row["pixel_count"] for row in subset) > 0
        for row in subset:
            assert row["area_percent"] == pytest.approx(20.0, abs=1.0)

    with rasterio.open(index_dir / "mb_ndvi_relative_5class.tif") as src:
        classes = src.read(1)
    mask = _valid_mask(multi_block_ortho.raster, multi_block_ortho.boundary)
    # NoData outside the field, and every valid pixel carries a class.
    assert np.all(classes[~mask] == 0)
    assert set(np.unique(classes[mask])) <= {1, 2, 3, 4, 5}


def test_prescription_totals_survive_multiple_windows(
    tmp_path: Path, multi_block_ortho
) -> None:
    from m3m.prescription import RateSchedule, build

    index_dir = tmp_path / "indices"
    indices.calculate(
        multi_block_ortho.raster, multi_block_ortho.boundary, index_dir, prefix="mb"
    )
    zones.run(index_dir, multi_block_ortho.boundary, prefix="mb", make_figures=False)

    result = build(
        index_dir / "mb_ndvi_relative_5class.tif",
        output_dir=tmp_path / "rx",
        prefix="mb",
        schedule=RateSchedule({1: 0.0, 2: 40.0, 3: 80.0, 4: 120.0, 5: 160.0}, unit="kg N/ha"),
        target_mean_rate=120.0,
        make_figure=False,
    )
    mask = _valid_mask(multi_block_ortho.raster, multi_block_ortho.boundary)
    assert result.totals["valid_pixels"] == int(mask.sum())
    with rasterio.open(result.rate_raster) as src:
        rates = src.read(1)
    valid = rates != -9999.0
    assert int(valid.sum()) == int(mask.sum())
    assert float(rates[valid].mean()) == pytest.approx(120.0, rel=1e-3)
