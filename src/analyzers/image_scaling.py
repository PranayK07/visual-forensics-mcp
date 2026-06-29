"""Image scaling / stretch / effective-DPI computations.

Given an embedded image's original pixel dimensions and its displayed size on
the page (in PDF points), compute:

* ``scale_x`` / ``scale_y`` -- points consumed per source pixel on each axis.
* ``effective_dpi_x`` / ``effective_dpi_y`` -- the true rendered resolution.
* ``stretch_ratio`` -- normalised difference between the two axis scales.

These are pure functions; the detector layer applies thresholds.
"""

from __future__ import annotations

from typing import Any

POINTS_PER_INCH = 72.0


def compute_scaling(
    original_width: int,
    original_height: int,
    display_width_pts: float,
    display_height_pts: float,
) -> dict[str, float]:
    ow = max(int(original_width), 1)
    oh = max(int(original_height), 1)
    dw = float(display_width_pts)
    dh = float(display_height_pts)

    scale_x = dw / ow if ow else 0.0
    scale_y = dh / oh if oh else 0.0

    eff_dpi_x = (ow * POINTS_PER_INCH / dw) if dw > 1e-6 else 0.0
    eff_dpi_y = (oh * POINTS_PER_INCH / dh) if dh > 1e-6 else 0.0

    denom = max(abs(scale_x), abs(scale_y), 1e-9)
    stretch_ratio = abs(scale_x - scale_y) / denom

    return {
        "scale_x": scale_x,
        "scale_y": scale_y,
        "effective_dpi_x": eff_dpi_x,
        "effective_dpi_y": eff_dpi_y,
        "stretch_ratio": stretch_ratio,
    }
