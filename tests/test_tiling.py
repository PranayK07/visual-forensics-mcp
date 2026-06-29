"""Tile generation tests."""

from __future__ import annotations

import numpy as np

from src.tiling import generate_tiles
from src.utils.config import load_config


def _gradient_image(h: int, w: int) -> np.ndarray:
    x = np.linspace(0, 255, w, dtype=np.uint8)
    return np.tile(x, (h, 1))


def test_tiles_cover_page_with_overlap():
    config = load_config(
        overrides={"tiling": {"tile_size": 256, "tile_overlap": 0.25}}
    )
    img = _gradient_image(800, 700)
    rgb = np.stack([img] * 3, axis=-1)
    tiles = generate_tiles(1, img, rgb, config)

    assert len(tiles) > 1
    # tile ids are sequential and unique per page
    ids = [t.tile_id for t in tiles]
    assert ids == list(range(len(tiles)))

    for t in tiles:
        x1, y1, x2, y2 = t.bbox
        assert 0 <= x1 < x2 <= 700
        assert 0 <= y1 < y2 <= 800
        assert t.page == 1
        assert t.image.shape[0] == int(y2 - y1)
        assert t.image.shape[1] == int(x2 - x1)

    # Overlap check: with 25% overlap, second column starts before first ends.
    xs = sorted({t.bbox[0] for t in tiles})
    if len(xs) >= 2:
        step = xs[1] - xs[0]
        assert step < 256  # less than tile size => overlapping


def test_blank_tile_flagged():
    config = load_config(overrides={"tiling": {"tile_size": 128}})
    img = np.full((300, 300), 255, dtype=np.uint8)  # uniform -> blank
    rgb = np.stack([img] * 3, axis=-1)
    tiles = generate_tiles(2, img, rgb, config)
    assert tiles
    assert all(t.blank for t in tiles)
