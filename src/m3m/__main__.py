"""Allow ``python -m m3m`` as an equivalent of the ``m3m`` console script.

This matters beyond convenience: the generated OpenDroneMap script calls
``python -m m3m fill-nodata`` inside the ODM container, where the console script
is not installed but the package directory is mounted on ``PYTHONPATH``.
"""

from __future__ import annotations

from m3m.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
