"""The synthetic generator and the XMP reader must agree by construction."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from m3m import synthetic
from m3m.constants import BANDS
from m3m.io import (
    REQUIRED_XMP_TAGS,
    extract_xmp_packet,
    find_band,
    missing_xmp_tags,
    read_dji_xmp,
    scan_capture_groups,
)

ROI = (4, 6, 20, 24)
SHAPE = (48, 64)


def test_xmp_written_by_the_generator_is_read_back_completely(tmp_path: Path) -> None:
    dataset = synthetic.generate_raw_dataset(
        tmp_path, groups=1, shape=SHAPE, panel_roi=ROI, with_rgb=False
    )
    path = find_band(dataset.img_dir, "NIR")[0]
    metadata = read_dji_xmp(path)

    assert missing_xmp_tags(metadata) == []
    assert metadata["BandName"] == "NIR"
    assert len(metadata["VignettingData"].split(",")) == 6
    assert set(REQUIRED_XMP_TAGS) <= set(metadata)


@pytest.mark.parametrize(
    "raw",
    [
        # How DJI writes the packet.
        '<x:xmpmeta xmlns:x="adobe:ns:meta/"></x:xmpmeta>',
        # How GDAL serialises an xml:* metadata domain entry.
        'xml:XMP=<x:xmpmeta xmlns:x="adobe:ns:meta/"></x:xmpmeta>',
        # Extra whitespace / a different domain path.
        '  rdf:RDF=<x:xmpmeta xmlns:x="adobe:ns:meta/"></x:xmpmeta>\n',
    ],
)
def test_extract_xmp_packet_accepts_both_storage_forms(raw: str) -> None:
    assert extract_xmp_packet(raw) == '<x:xmpmeta xmlns:x="adobe:ns:meta/"></x:xmpmeta>'


@pytest.mark.parametrize("raw", [None, "", "no xml here"])
def test_extract_xmp_packet_returns_none_without_a_packet(raw: str | None) -> None:
    assert extract_xmp_packet(raw) is None


def test_scan_capture_groups_pairs_bands_with_the_rgb_frame(tmp_path: Path) -> None:
    dataset = synthetic.generate_raw_dataset(
        tmp_path, groups=3, shape=SHAPE, panel_roi=ROI, with_rgb=True
    )
    groups = scan_capture_groups(dataset.img_dir)

    assert sorted(groups) == ["0001", "0002", "0003"]
    for paths in groups.values():
        assert set(paths) == {*BANDS, "RGB"}
        assert paths["RGB"].suffix.lower() == ".jpg"


def test_duplicate_band_frame_is_rejected(tmp_path: Path) -> None:
    dataset = synthetic.generate_raw_dataset(
        tmp_path, groups=1, shape=SHAPE, panel_roi=ROI, with_rgb=False
    )
    source = find_band(dataset.img_dir, "NIR")[0]
    (dataset.img_dir / "synthetic_0001_MS_NIR_copy_MS_NIR.TIF").write_bytes(
        source.read_bytes()
    )
    with pytest.raises(ValueError, match="Duplicate NIR frame"):
        scan_capture_groups(dataset.img_dir)


def test_find_band_is_case_insensitive(tmp_path: Path) -> None:
    dataset = synthetic.generate_raw_dataset(
        tmp_path, groups=1, shape=SHAPE, panel_roi=ROI, with_rgb=False
    )
    assert find_band(dataset.img_dir, "nir") == find_band(dataset.img_dir, "NIR")


def test_panel_roi_outside_the_image_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="does not fit inside"):
        synthetic.generate_raw_dataset(
            tmp_path, groups=1, shape=(16, 16), panel_roi=(10, 10, 40, 40), with_rgb=False
        )


def test_reflectance_that_would_overflow_uint16_is_rejected() -> None:
    with pytest.raises(ValueError, match="overflow uint16"):
        synthetic.dn_from_reflectance(
            np.full((4, 4), 2.0),
            scale=1.0,
            vignette=np.ones((4, 4)),
        )


def test_generation_is_deterministic(tmp_path: Path) -> None:
    first = synthetic.generate_orthomosaic(tmp_path / "a", shape=SHAPE)
    second = synthetic.generate_orthomosaic(tmp_path / "b", shape=SHAPE)
    assert first.raster.read_bytes() == second.raster.read_bytes()


def test_generated_config_resolves_relative_to_itself(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression: a relative ``--dest`` used to produce ``demo/demo/Dataset``.

    The config records paths relative to its own directory, so resolving it from
    an unrelated working directory must still find the generated data.
    """
    from m3m.config import load_settings

    dest = tmp_path / "demo"
    dataset = synthetic.generate_raw_dataset(
        dest, groups=1, shape=SHAPE, panel_roi=ROI, with_rgb=False
    )
    ortho = synthetic.generate_orthomosaic(dest, shape=SHAPE)
    config_path = synthetic.write_config(dest / "m3m.toml", dataset, orthomosaic=ortho)

    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)

    settings = load_settings(config_path, env={})
    assert settings.paths.img_dir == dataset.img_dir.resolve()
    assert settings.paths.panel_dir == dataset.panel_dir.resolve()
    assert settings.paths.boundary == ortho.boundary.resolve()
    # Every input path must actually resolve to something that exists.
    assert settings.paths.require("img_dir", flag="--img-dir").is_dir()
    assert settings.paths.require("boundary", flag="--boundary").is_file()


def test_config_survives_being_moved_with_its_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from m3m.config import load_settings

    original = tmp_path / "original"
    dataset = synthetic.generate_raw_dataset(
        original, groups=1, shape=SHAPE, panel_roi=ROI, with_rgb=False
    )
    ortho = synthetic.generate_orthomosaic(original, shape=SHAPE)
    synthetic.write_config(original / "m3m.toml", dataset, orthomosaic=ortho)

    moved = tmp_path / "moved"
    original.rename(moved)
    monkeypatch.chdir(tmp_path)

    settings = load_settings(moved / "m3m.toml", env={})
    assert settings.paths.require("img_dir", flag="--img-dir").is_dir()
    assert settings.paths.require("boundary", flag="--boundary").is_file()
