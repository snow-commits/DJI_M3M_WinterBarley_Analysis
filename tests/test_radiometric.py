"""The radiometric model, including the round trip it must invert exactly."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from m3m import radiometric, synthetic
from m3m.constants import BANDS
from m3m.io import find_band, read_dji_xmp, read_raster

ROI = (4, 6, 20, 24)
SHAPE = (48, 64)


def _anchor(dataset: synthetic.RawDataset) -> radiometric.PanelAnchor:
    return radiometric.calibrate_nir_anchor(
        panel_dir=dataset.panel_dir,
        panel_roi={band: dataset.panel_roi for band in BANDS},
        panel_reflectance=dataset.panel_reflectance,
        saturation_dn=dataset.saturation_dn,
    )


def test_signal_model_rejects_non_integer_input() -> None:
    with pytest.raises(radiometric.RadiometricError, match="integer digital numbers"):
        radiometric.dji_relative_signal(
            np.zeros((4, 4), dtype=np.float32), {"BlackLevel": "0"}
        )


def test_anchor_recovers_the_generator_scale(tmp_path: Path) -> None:
    dataset = synthetic.generate_raw_dataset(
        tmp_path, groups=1, shape=SHAPE, panel_roi=ROI, scene_scale=0.8, with_rgb=False
    )
    anchor = _anchor(dataset)
    assert anchor.scale == pytest.approx(0.8, rel=1e-4)
    assert anchor.valid_fraction == 1.0


def test_calibrated_reflectance_matches_the_generated_target(tmp_path: Path) -> None:
    """Inverting the DJI chain must return the reflectance we asked for."""
    dataset = synthetic.generate_raw_dataset(
        tmp_path, groups=2, shape=SHAPE, panel_roi=ROI, with_rgb=False
    )
    anchor = _anchor(dataset)
    groups = radiometric.collect_flight_groups(dataset.img_dir)

    assert dataset.target_reflectance is not None
    for group_id, paths in groups.items():
        for band in BANDS:
            data, _ = read_raster(paths[band])
            metadata = read_dji_xmp(paths[band])
            reflectance = radiometric.dji_relative_signal(data, metadata) * anchor.scale
            expected = dataset.target_reflectance[band]
            assert np.allclose(reflectance, expected, atol=2e-4), (
                f"{group_id}/{band} round trip drifted"
            )


def test_vignetting_round_trips_with_nonzero_coefficients(tmp_path: Path) -> None:
    """A real vignetting polynomial must also be inverted correctly."""
    coefficients = (1e-4, 1e-7, 0.0, 0.0, 0.0, 0.0)
    dataset = synthetic.generate_raw_dataset(
        tmp_path,
        groups=1,
        shape=SHAPE,
        panel_roi=ROI,
        vignette_coefficients=coefficients,
        with_rgb=False,
    )
    anchor = _anchor(dataset)
    groups = radiometric.collect_flight_groups(dataset.img_dir)
    paths = groups["0001"]

    data, _ = read_raster(paths["NIR"])
    metadata = read_dji_xmp(paths["NIR"])
    reflectance = radiometric.dji_relative_signal(data, metadata) * anchor.scale

    assert dataset.target_reflectance is not None
    assert np.allclose(reflectance, dataset.target_reflectance["NIR"], atol=5e-4)


def test_vignetting_correction_brightens_the_corners() -> None:
    """The polynomial must raise corner signal relative to the centre."""
    shape = (64, 64)
    coefficients = (1e-3, 0.0, 0.0, 0.0, 0.0, 0.0)
    rows, columns = np.indices(shape, dtype=np.float64)
    radius = np.hypot(columns - 32.0, rows - 32.0)
    factor = 1.0 + coefficients[0] * radius
    assert factor[0, 0] > factor[32, 32]

    # Feeding a flat capture through the model must yield a flat-relative result
    # only after the vignette is divided out; here we assert the direction.
    metadata = {
        "BlackLevel": "0",
        "SensorGain": "1",
        "ExposureTime": "1000",
        "SensorGainAdjustment": "1",
        "Irradiance": "1",
        "VignettingData": ",".join(str(value) for value in coefficients),
        "CalibratedOpticalCenterX": "32",
        "CalibratedOpticalCenterY": "32",
    }
    flat = np.full(shape, 10000, dtype=np.uint16)
    corrected = radiometric.dji_relative_signal(flat, metadata)
    assert corrected[0, 0] > corrected[32, 32]


def test_saturated_pixels_become_nodata(tmp_path: Path) -> None:
    """Saturated pixels must be excluded and written as the NoData sentinel."""
    import rasterio

    from m3m.constants import NODATA

    dataset = synthetic.generate_raw_dataset(
        tmp_path, groups=1, shape=SHAPE, panel_roi=ROI, with_rgb=False
    )
    anchor = _anchor(dataset)
    groups = radiometric.collect_flight_groups(dataset.img_dir)
    path = groups["0001"]["NIR"]

    # Blow out the top rows so part of the capture saturates.
    with rasterio.open(path, "r+") as src:
        data = src.read(1)
        data[:8, :] = 65535
        src.write(data, 1)

    result = radiometric.calibrate_and_write(
        path,
        "NIR",
        anchor.scale,
        tmp_path / "out",
        saturation_dn=dataset.saturation_dn,
        panel_reflectance_nir=anchor.panel_reflectance_nir,
    )
    assert 0.0 < result["saturated_fraction"] < 1.0

    with rasterio.open(result["output"]) as src:
        written = src.read(1)
        assert src.nodata == NODATA
    assert np.all(written[:8, :] == NODATA)
    assert np.all(written[8:, :] != NODATA)


def test_fully_saturated_capture_is_rejected(tmp_path: Path) -> None:
    dataset = synthetic.generate_raw_dataset(
        tmp_path, groups=1, shape=SHAPE, panel_roi=ROI, with_rgb=False
    )
    anchor = _anchor(dataset)
    groups = radiometric.collect_flight_groups(dataset.img_dir)
    with pytest.raises(radiometric.RadiometricError, match="no usable pixels"):
        radiometric.calibrate_and_write(
            groups["0001"]["NIR"],
            "NIR",
            anchor.scale,
            tmp_path / "out",
            saturation_dn=0,
            panel_reflectance_nir=anchor.panel_reflectance_nir,
        )


def test_incomplete_group_is_reported_not_silently_skipped(tmp_path: Path) -> None:
    dataset = synthetic.generate_raw_dataset(
        tmp_path, groups=2, shape=SHAPE, panel_roi=ROI, with_rgb=False
    )
    find_band(dataset.img_dir, "RE")[0].unlink()
    with pytest.raises(ValueError, match="missing spectral bands"):
        radiometric.collect_flight_groups(dataset.img_dir)


def test_fully_saturated_panel_roi_is_rejected(tmp_path: Path) -> None:
    dataset = synthetic.generate_raw_dataset(
        tmp_path, groups=1, shape=SHAPE, panel_roi=ROI, with_rgb=False
    )
    with pytest.raises(radiometric.RadiometricError, match="saturated or invalid"):
        radiometric.calibrate_nir_anchor(
            panel_dir=dataset.panel_dir,
            panel_roi={band: dataset.panel_roi for band in BANDS},
            panel_reflectance=dataset.panel_reflectance,
            saturation_dn=1,
        )


def test_capture_without_xmp_is_rejected_with_a_clear_message(tmp_path: Path) -> None:
    import rasterio

    from m3m.io import REQUIRED_XMP_TAGS, missing_xmp_tags, read_dji_xmp

    bare = tmp_path / "bare_0001_MS_NIR.TIF"
    with rasterio.open(
        bare,
        "w",
        driver="GTiff",
        count=1,
        dtype="uint16",
        width=8,
        height=8,
    ) as dst:
        dst.write(np.full((8, 8), 1000, dtype=np.uint16), 1)

    assert read_dji_xmp(bare) == {}
    assert missing_xmp_tags({}) == list(REQUIRED_XMP_TAGS)

    with pytest.raises(radiometric.RadiometricError, match="XMP is missing"):
        radiometric.calibrate_and_write(
            bare,
            "NIR",
            1.0,
            tmp_path / "out",
            saturation_dn=65520,
            panel_reflectance_nir=0.493,
        )


def test_roi_overlay_is_written_headlessly(tmp_path: Path) -> None:
    dataset = synthetic.generate_raw_dataset(
        tmp_path, groups=1, shape=SHAPE, panel_roi=ROI, with_rgb=False
    )
    path = find_band(dataset.panel_dir, "NIR")[0]
    output = radiometric.write_roi_overlay(path, ROI, tmp_path / "roi.png")
    assert output.is_file() and output.stat().st_size > 0
