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
        dst.set_band_description(1, "Green")  # the default map says Red is here
    with pytest.raises(ValueError, match="band map says 'Red' is there"):
        indices.calculate(wrong, orthomosaic.boundary, tmp_path / "out")


def test_band_map_pointing_past_the_raster_is_rejected(tmp_path: Path, orthomosaic) -> None:
    small = tmp_path / "three_bands.tif"
    with rasterio.open(orthomosaic.raster) as src:
        profile = src.profile.copy()
        data = src.read((1, 2, 3))
    profile.update(count=3)
    with rasterio.open(small, "w", **profile) as dst:
        dst.write(data)
    with pytest.raises(ValueError, match="points past the end of the raster"):
        indices.calculate(small, orthomosaic.boundary, tmp_path / "out")


# --------------------------------------------------------------------------- #
# configurable band layout
# --------------------------------------------------------------------------- #


def _write_micasense_style(tmp_path: Path, orthomosaic) -> Path:
    """Re-stack the fixture as Blue, Green, Red, RedEdge, NIR with no alpha."""
    target = tmp_path / "micasense_order.tif"
    with rasterio.open(orthomosaic.raster) as src:
        profile = src.profile.copy()
        red = src.read(1)
        green = src.read(2)
        nir = src.read(3)
        rededge = src.read(4)
    with rasterio.open(target, "w", **profile) as dst:
        for position, (name, band) in enumerate(
            (
                ("Blue", (red * 0.6).astype("float32")),
                ("Green", green),
                ("Red", red),
                ("RedEdge", rededge),
                ("NIR", nir),
            ),
            start=1,
        ):
            dst.write(band, position)
            dst.set_band_description(position, name)
    return target


MICASENSE_BAND_MAP = {"blue": 1, "green": 2, "red": 3, "rededge": 4, "nir": 5, "alpha": 0}


def test_micasense_style_layout_is_supported(tmp_path: Path, orthomosaic) -> None:
    """A different band order and no alpha band must work when described."""
    raster = _write_micasense_style(tmp_path, orthomosaic)
    output_dir = tmp_path / "out"
    rows = indices.calculate(
        raster, orthomosaic.boundary, output_dir, prefix="ms", band_map=MICASENSE_BAND_MAP
    )

    with rasterio.open(raster) as src:
        red = src.read(3)
        nir = src.read(5)
    with rasterio.open(output_dir / "ms_ndvi.tif") as src:
        ndvi = src.read(1)
        tags = src.tags(1)

    mask = ndvi != NODATA
    assert mask.any()
    assert np.allclose(ndvi[mask], ((nir - red) / (nir + red))[mask], atol=1e-5)
    with rasterio.open(output_dir / "ms_ndre.tif") as src:
        ndre = src.read(1)
    with rasterio.open(raster) as src:
        rededge = src.read(4)
    assert np.allclose(
        ndre[mask], ((nir - rededge) / (nir + rededge))[mask], atol=1e-5
    )
    assert np.allclose(
        np.array([row["valid_pixels"] for row in rows]), rows[0]["valid_pixels"]
    )
    # The metadata must record the layout actually used, not the default one.
    assert "Red=3" in tags["BAND_MAPPING"]
    assert "NIR=5" in tags["BAND_MAPPING"]
    assert "Alpha" not in tags["BAND_MAPPING"]
    assert "no alpha band configured" in tags["MASK"]


def test_default_map_and_explicit_default_agree(tmp_path: Path, orthomosaic) -> None:
    """Passing the default map explicitly must change nothing."""
    from m3m.constants import DEFAULT_ORTHOMOSAIC_BAND_MAP

    first = tmp_path / "a"
    second = tmp_path / "b"
    indices.calculate(orthomosaic.raster, orthomosaic.boundary, first, prefix="d")
    indices.calculate(
        orthomosaic.raster,
        orthomosaic.boundary,
        second,
        prefix="d",
        band_map=DEFAULT_ORTHOMOSAIC_BAND_MAP,
    )
    assert (first / "d_ndvi.tif").read_bytes() == (second / "d_ndvi.tif").read_bytes()


def test_wrong_band_map_is_caught_when_the_export_is_labelled(
    tmp_path: Path, orthomosaic
) -> None:
    """The description cross-check is what stops a swapped map producing nonsense."""
    with pytest.raises(ValueError, match="band map says 'NIR' is there"):
        indices.calculate(
            orthomosaic.raster,
            orthomosaic.boundary,
            tmp_path / "out",
            band_map={"red": 3, "green": 2, "nir": 1, "rededge": 4, "alpha": 5},
        )


def test_wrong_band_map_is_undetectable_without_band_descriptions(
    tmp_path: Path, orthomosaic
) -> None:
    """Documents a real limit: with no descriptions there is nothing to check against.

    This is why the toolkit validates the layout against band descriptions and why
    the README asks you to set them on export.
    """
    unlabelled = tmp_path / "unlabelled.tif"
    with rasterio.open(orthomosaic.raster) as src:
        profile = src.profile.copy()
        data = src.read()
    with rasterio.open(unlabelled, "w", **profile) as dst:
        dst.write(data)  # descriptions deliberately not set

    good = tmp_path / "good"
    swapped = tmp_path / "swapped"
    indices.calculate(unlabelled, orthomosaic.boundary, good, prefix="u")
    indices.calculate(
        unlabelled,
        orthomosaic.boundary,
        swapped,
        prefix="u",
        band_map={"red": 3, "green": 2, "nir": 1, "rededge": 4, "alpha": 5},
    )
    with rasterio.open(good / "u_ndvi.tif") as src:
        a = src.read(1)
    with rasterio.open(swapped / "u_ndvi.tif") as src:
        b = src.read(1)
    # Both "succeed", and they disagree. Swapping red and NIR negates NDVI
    # exactly, so the wrong one is a plausible-looking map that only a human can
    # spot - which is precisely why the toolkit asks for band descriptions.
    mask = a != NODATA
    assert mask.any()
    assert np.allclose(a[mask], -b[mask], atol=1e-5)


def test_duplicate_band_positions_are_rejected(tmp_path: Path, orthomosaic) -> None:
    with pytest.raises(ValueError, match="both map to position"):
        indices.calculate(
            orthomosaic.raster,
            orthomosaic.boundary,
            tmp_path / "out",
            band_map={"red": 1, "nir": 1, "rededge": 4},
        )


def test_missing_required_band_in_map_is_rejected(tmp_path: Path, orthomosaic) -> None:
    with pytest.raises(ValueError, match="does not provide nir"):
        indices.calculate(
            orthomosaic.raster,
            orthomosaic.boundary,
            tmp_path / "out",
            band_map={"red": 1, "nir": 0, "rededge": 4},
        )


def test_unknown_band_name_in_map_is_rejected(tmp_path: Path, orthomosaic) -> None:
    with pytest.raises(ValueError, match="Unknown band name"):
        indices.calculate(
            orthomosaic.raster,
            orthomosaic.boundary,
            tmp_path / "out",
            band_map={"red": 1, "nir": 3, "rededge": 4, "swir": 2},
        )


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
