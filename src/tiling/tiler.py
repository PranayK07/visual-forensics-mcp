"""Split a rendered page into overlapping tiles.

Tiles are produced left-to-right, top-to-bottom with a configurable size and
fractional overlap. Each tile records its page, a stable per-page tile id, and
its pixel bounding box ``[x1, y1, x2, y2]``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..utils.config import Config


@dataclass
class Tile:
    """One image tile plus its metadata."""

    page: int
    tile_id: int
    bbox: list[float]  # [x1, y1, x2, y2] in page pixel coordinates
    image: np.ndarray  # grayscale crop
    image_rgb: np.ndarray  # rgb crop (for OCR / colour)
    blank: bool = False


def _axis_starts(length: int, tile: int, step: int, min_keep: int) -> list[int]:
    """Compute tile start offsets along one axis."""
    if length <= tile:
        return [0]
    starts: list[int] = []
    pos = 0
    while True:
        if pos + tile >= length:
            last = length - tile
            if not starts or last - starts[-1] >= min_keep:
                starts.append(last)
            break
        starts.append(pos)
        pos += step
    # de-duplicate while preserving order
    seen: set[int] = set()
    unique: list[int] = []
    for s in starts:
        s = max(0, s)
        if s not in seen:
            seen.add(s)
            unique.append(s)
    return unique


def generate_tiles(
    page: int,
    image_gray: np.ndarray,
    image_rgb: np.ndarray,
    config: Config,
) -> list[Tile]:
    """Generate overlapping tiles for a single page."""
    tile_size = int(config.get("tiling.tile_size", 512))
    overlap = float(config.get("tiling.tile_overlap", 0.20))
    min_fraction = float(config.get("tiling.min_tile_fraction", 0.30))
    blank_std = float(config.get("tiling.blank_std_threshold", 3.0))

    overlap = min(max(overlap, 0.0), 0.95)
    step = max(1, int(round(tile_size * (1.0 - overlap))))
    min_keep = max(1, int(round(tile_size * min_fraction)))

    height, width = image_gray.shape[:2]
    x_starts = _axis_starts(width, tile_size, step, min_keep)
    y_starts = _axis_starts(height, tile_size, step, min_keep)

    tiles: list[Tile] = []
    tile_id = 0
    for y0 in y_starts:
        y1 = min(y0 + tile_size, height)
        for x0 in x_starts:
            x1 = min(x0 + tile_size, width)
            crop_gray = image_gray[y0:y1, x0:x1]
            crop_rgb = image_rgb[y0:y1, x0:x1]
            if crop_gray.size == 0:
                continue
            blank = float(np.std(crop_gray)) < blank_std
            tiles.append(
                Tile(
                    page=page,
                    tile_id=tile_id,
                    bbox=[float(x0), float(y0), float(x1), float(y1)],
                    image=crop_gray,
                    image_rgb=crop_rgb,
                    blank=blank,
                )
            )
            tile_id += 1

    return tiles
