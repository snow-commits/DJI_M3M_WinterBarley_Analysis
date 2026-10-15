"""Index maths and the memory-bounded statistics that summarise them."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import rasterio

from m3m import indices, zones
from m3m.constants import INDEX_RANGES, NODATA


def test_index_stats_quantiles_match_numpy() -> None:
    rng = np.random.default_rng(7)
    values = rng.uniform(-0.9, 0.9, size=50_000)
    stats = indices.IndexStats(name="NDVI", value_range=(-1.0, 1.0))
    stats.add(values)

    assert stats.count == values.size
    assert stats.mean == pytest.approx(float(values.mean()), rel=1e-9)
    assert stats.stddev == pytest.approx(float(values.std()), rel=1e-3)
    assert stats.minimum == pytest.approx(float(values.min()))
    assert stats.maximum == pytest.approx(float(values.max()))
    # The documented method is a 100k-bin histogram midpoint estimate, so the
    # quantile is accurate to roughly the spacing of adjacent order statistics
    # (~1.8 bins for this sample size), not to the bin width itself.
    assert stats.quantile(0.5) == pytest.approx(float(np.median(values)), abs=2e-4)
    assert stats.quantile(0.95) == pytest.approx(
        float(np.percentile(values, 95)), abs=2e-4
    )


def test_stats_ignore_empty_windows() -> None:
    stats = indices.IndexStats(name="NDVI", value_range=(-1.0, 1.0))
    stats.add(np.array([], dtype=np.float64))
    assert stats.count == 0
    assert np.isnan(stats.mean)
    assert np.isnan(stats.quantile(0.5))


def test_index_rasters_match_a_direct_numpy_computation(
    tmp_path: Path, orthomosaic
) -> None:
    """Independent recomputation from the written bands must agree."""
    output_dir = tmp_path / "indices"
    indices.calculate(
        orthomosaic.raster, orthomosaic.boundary, output_dir, prefix="t"
    )

    with rasterio.open(orthomosaic.raster) as src:
        red = src.read(1)
        nir = src.read(3)
        rededge = src.read(4)
        alpha = src.read(5)

    inside = np.ones(red.shape, dtype=bool)
    valid = inside & (alpha > 0) & (red >= 0) & (red <= 1)
    with rasterio.open(output_dir / "t_ndvi.tif") as src:
        ndvi = src.read(1)
    with rasterio.open(output_dir / "t_ndre.tif") as src:
        ndre = src.read(1)
    with rasterio.open(output_dir / "t_evi2.tif") as src:
        evi2 = src.read(1)

    expected_ndvi = (nir - red) / (nir + red)
    expected_ndre = (nir - rededge) / (nir + rededge)
    expected_evi2 = 2.5 * (nir - red) / (nir + 2.4 * red + 1)

    mask = valid & (ndvi != NODATA)
    assert mask.any()
    assert np.allclose(ndvi[mask], expected_ndvi[mask], atol=1e-5)
    assert np.allclose(ndre[mask], expected_ndre[mask], atol=1e-5)
    assert np.allclose(evi2[mask], expected_evi2[mask], atol=1e-5)
    # Outside the field boundary the index rasters must stay NoData.
    assert np.all(ndvi[~mask] == NODATA)


def test_out_of_range_reflectance_is_masked_out(tmp_path: Path, orthomosaic) -> None:
    """Reflectance above 1 is physically impossible and must not enter the stats."""
    damaged = tmp_path / "damaged.tif"
    with rasterio.open(orthomosaic.raster) as src:
        profile = src.profile.copy()
        data = src.read()
        descriptions = src.descriptions
    # Damage only the top rows: some pixels must remain valid, otherwise the
    # command is expected to fail loudly instead of reporting statistics.
    data[0, :20, :] = 1.5  # Red reflectance out of range
    with rasterio.open(damaged, "w", **profile) as dst:
        dst.write(data)
        for position, name in enumerate(descriptions, start=1):
            if name:
                dst.set_band_description(position, name)

    rows = indices.calculate(damaged, orthomosaic.boundary, tmp_path / "out", prefix="d")
    ndvi_row = next(row for row in rows if row["index"] == "NDVI")
    with rasterio.open(orthomosaic.raster) as src:
        height, width = src.height, src.width
    assert 0 < ndvi_row["valid_pixels"] < height * width


def test_band_description_mismatch_is_rejected(tmp_path: Path, orthomosaic) -> None:
    wrong = tmp_path / "wrong_bands.tif"
    with rasterio.open(orthomosaic.raster) as src:
        profile = src.profile.copy()
        data = src.read()
    with rasterio.open(wrong, "w", **profile) as dst:
        dst.write(data)
        dst.set_band_description(1, "Green")  # should be Red
    with pytest.raises(ValueError, match="should be 'Red'"):
        indices.calculate(wrong, orthomosaic.boundary, tmp_path / "out")


def test_too_few_bands_is_rejected(tmp_path: Path, orthomosaic) -> None:
    small = tmp_path / "three_bands.tif"
    with rasterio.open(orthomosaic.raster) as src:
        profile = src.profile.copy()
        data = src.read((1, 2, 3))
    profile.update(count=3)
    with rasterio.open(small, "w", **profile) as dst:
        dst.write(data)
    with pytest.raises(ValueError, match="four reflectance bands plus an alpha band"):
        indices.calculate(small, orthomosaic.boundary, tmp_path / "out")


def test_unknown_index_name_is_rejected(tmp_path: Path, orthomosaic) -> None:
    with pytest.raises(ValueError, match="Unknown index name"):
        indices.calculate(
            orthomosaic.raster,
            orthomosaic.boundary,
            tmp_path / "out",
            index_names=("NDWI",),
        )


def test_zones_are_equal_frequency_per_index(tmp_path: Path, orthomosaic) -> None:
    index_dir = tmp_path / "indices"
    indices.calculate(orthomosaic.raster, orthomosaic.boundary, index_dir, prefix="z")
    rows = zones.run(
        index_dir, orthomosaic.boundary, prefix="z", make_figures=False
    )

    for name in ("NDVI", "NDRE", "EVI2"):
        subset = [row for row in rows if row["index"] == name]
        assert len(subset) == 5
        # Equal-frequency quantiles: every class should hold roughly a fifth.
        for row in subset:
            assert row["area_percent"] == pytest.approx(20.0, abs=2.0)
        assert sum(row["pixel_count"] for row in subset) > 0


def test_zone_thresholds_are_monotonic(tmp_path: Path, orthomosaic) -> None:
    index_dir = tmp_path / "indices"
    indices.calculate(orthomosaic.raster, orthomosaic.boundary, index_dir, prefix="m")
    thresholds = zones.calculate_thresholds(index_dir / "m_ndvi.tif", "ndvi")
    assert thresholds == sorted(thresholds)
    low, high = INDEX_RANGES["ndvi"]
    assert all(low <= value <= high for value in thresholds)


def test_classify_writes_nodata_outside_the_field(tmp_path: Path, orthomosaic) -> None:
    index_dir = tmp_path / "indices"
    indices.calculate(orthomosaic.raster, orthomosaic.boundary, index_dir, prefix="c")
    zones.run(index_dir, orthomosaic.boundary, prefix="c", make_figures=False)
    with rasterio.open(index_dir / "c_ndvi_relative_5class.tif") as src:
        classes = src.read(1)
        assert src.nodata == 0
    assert set(np.unique(classes)) <= {0, 1, 2, 3, 4, 5}
    assert (classes == 0).any()
