"""Shared constants for the M3M toolkit.

Keeping the band order, the NoData sentinel and the index definitions in one
place avoids the drift that occurs when three scripts each carry their own copy
of the same formula.
"""

from __future__ import annotations

#: Multispectral bands produced by the DJI Mavic 3 Multispectral camera.
#: The camera has no independent blue band, so blue-dependent indices such as
#: standard EVI cannot be computed from M3M data.
BANDS: tuple[str, ...] = ("G", "R", "RE", "NIR")

#: Band name used for the blue channel in RGB imagery.
RGB_BAND = "RGB"

#: Sentinel written into float32 outputs for saturated / masked pixels.
NODATA: float = -9999.0

#: Guard against division by near-zero denominators in index formulas.
EPSILON: float = 1e-8

#: Number of histogram bins used for all memory-bounded quantile estimates.
HISTOGRAM_BINS: int = 100_000

#: Vegetation indices computed by this toolkit.
#:
#: ``histogram_range`` must bracket the full theoretical range of the index so
#: that streaming histograms do not silently clip values.
INDEX_SPECS: dict[str, dict[str, object]] = {
    "NDVI": {
        "formula": "(NIR - Red) / (NIR + Red)",
        "histogram_range": (-1.0, 1.0),
    },
    "NDRE": {
        "formula": "(NIR - RedEdge) / (NIR + RedEdge)",
        "histogram_range": (-1.0, 1.0),
    },
    "EVI2": {
        "formula": "2.5 * (NIR - Red) / (NIR + 2.4 * Red + 1)",
        "histogram_range": (-2.5 / 3.4, 1.25),
    },
}

#: Lower-case index names in the canonical order used by zones and comparisons.
INDEX_ORDER: tuple[str, ...] = ("ndvi", "ndre", "evi2")

#: Histogram range per lower-case index name.
INDEX_RANGES: dict[str, tuple[float, float]] = {
    name.lower(): spec["histogram_range"]  # type: ignore[arg-type]
    for name, spec in INDEX_SPECS.items()
}

#: Human-readable names for the five within-field relative classes.
RELATIVE_CLASS_NAMES: tuple[str, ...] = (
    "Lowest relative",
    "Low",
    "Middle",
    "High",
    "Highest relative",
)
