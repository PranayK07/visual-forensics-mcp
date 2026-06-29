"""Shared helpers for detectors."""

from __future__ import annotations

from typing import Any, Callable, Optional

import numpy as np

from ..schemas.models import Finding, TileResult


def zscore_confidence(z: float, z_threshold: float) -> float:
    """Map a z-score magnitude to a 0-1 confidence.

    At ``|z| == z_threshold`` confidence is 0.5; at ``2 * z_threshold`` it
    saturates at 1.0. This is a monotonic, deterministic mapping.
    """
    if z_threshold <= 0:
        return 1.0 if abs(z) > 0 else 0.0
    return float(min(1.0, max(0.0, abs(z) / (2.0 * z_threshold))))


def collect_metric(
    tiles: list[TileResult],
    metric_key: str,
    text_density_min: float,
) -> tuple[list[int], np.ndarray]:
    """Return indices and values of tiles eligible for distribution analysis.

    Eligible tiles are non-blank, carry the requested metric (not None), and
    have at least ``text_density_min`` content.
    """
    idx: list[int] = []
    vals: list[float] = []
    for i, tile in enumerate(tiles):
        if tile.blank:
            continue
        value = tile.metrics.get(metric_key)
        if value is None:
            continue
        td = tile.metrics.get("text_density", 0.0) or 0.0
        if td < text_density_min:
            continue
        idx.append(i)
        vals.append(float(value))
    return idx, np.asarray(vals, dtype=np.float64)


def low_side_zscore_findings(
    tiles: list[TileResult],
    metric_key: str,
    *,
    page: int,
    finding_type: str,
    z_threshold: float,
    text_density_min: float,
    min_neighbors: int,
    absolute_floor: Optional[float],
    explanation: str,
    extra_metrics: Optional[Callable[[TileResult], dict[str, Any]]] = None,
) -> list[Finding]:
    """Flag tiles whose metric is anomalously LOW vs the page distribution."""
    idx, vals = collect_metric(tiles, metric_key, text_density_min)
    findings: list[Finding] = []
    if vals.size < min_neighbors:
        # Not enough context for a distribution; only absolute floor applies.
        if absolute_floor is not None:
            for i in idx:
                v = float(tiles[i].metrics.get(metric_key))
                if v < absolute_floor:
                    findings.append(
                        _make_finding(
                            tiles[i], metric_key, v, page, finding_type,
                            confidence=0.6, explanation=explanation,
                            extra_metrics=extra_metrics,
                            page_mean=None,
                        )
                    )
        return findings

    mean = float(vals.mean())
    std = float(vals.std())
    for local_i, tile_i in enumerate(idx):
        v = vals[local_i]
        z = (v - mean) / std if std > 1e-9 else 0.0
        is_low_z = z <= -z_threshold
        is_below_floor = absolute_floor is not None and v < absolute_floor
        if is_low_z or is_below_floor:
            conf = zscore_confidence(z, z_threshold) if is_low_z else 0.6
            findings.append(
                _make_finding(
                    tiles[tile_i], metric_key, v, page, finding_type,
                    confidence=conf, explanation=explanation,
                    extra_metrics=extra_metrics, page_mean=mean, zscore=z,
                )
            )
    return findings


def two_sided_zscore_findings(
    tiles: list[TileResult],
    metric_key: str,
    *,
    page: int,
    finding_type: str,
    z_threshold: float,
    text_density_min: float,
    min_neighbors: int,
    explanation: str,
) -> list[Finding]:
    """Flag tiles whose metric deviates strongly in EITHER direction."""
    idx, vals = collect_metric(tiles, metric_key, text_density_min)
    findings: list[Finding] = []
    if vals.size < min_neighbors:
        return findings
    mean = float(vals.mean())
    std = float(vals.std())
    if std <= 1e-9:
        return findings
    for local_i, tile_i in enumerate(idx):
        v = vals[local_i]
        z = (v - mean) / std
        if abs(z) >= z_threshold:
            findings.append(
                _make_finding(
                    tiles[tile_i], metric_key, v, page, finding_type,
                    confidence=zscore_confidence(z, z_threshold),
                    explanation=explanation, page_mean=mean, zscore=z,
                )
            )
    return findings


def _make_finding(
    tile: TileResult,
    metric_key: str,
    value: float,
    page: int,
    finding_type: str,
    *,
    confidence: float,
    explanation: str,
    extra_metrics: Optional[Callable[[TileResult], dict[str, Any]]] = None,
    page_mean: Optional[float] = None,
    zscore: Optional[float] = None,
) -> Finding:
    metrics: dict[str, Any] = {metric_key: round(float(value), 4)}
    if page_mean is not None:
        metrics["page_mean"] = round(float(page_mean), 4)
    if zscore is not None:
        metrics["zscore"] = round(float(zscore), 4)
    if "ocr_confidence" in tile.metrics and tile.metrics["ocr_confidence"] is not None:
        metrics.setdefault("ocr_confidence", tile.metrics["ocr_confidence"])
    if extra_metrics is not None:
        metrics.update(extra_metrics(tile))
    return Finding(
        type=finding_type,
        page=page,
        bbox=list(tile.bbox),
        metrics=metrics,
        confidence=round(float(confidence), 4),
        explanation=explanation,
    )
