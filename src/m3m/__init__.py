"""m3m - a label-free processing toolkit for DJI Mavic 3 Multispectral imagery.

The package deliberately keeps its import surface light: importing :mod:`m3m`
must not pull in ``rasterio``/``geopandas`` so that ``m3m --help`` and
``m3m --version`` stay fast and dependency-light.

Submodules are imported explicitly by the caller (or by :mod:`m3m.cli`).
"""

from __future__ import annotations

__version__ = "0.2.0"

#: Short project identifier used in generated report headers.
PROJECT_NAME = "m3m"

__all__ = ["PROJECT_NAME", "__version__"]
