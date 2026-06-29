"""Geometry helpers for bounding boxes (all in [x1, y1, x2, y2] form)."""

from __future__ import annotations


def area(bbox: list[float]) -> float:
    w = max(0.0, bbox[2] - bbox[0])
    h = max(0.0, bbox[3] - bbox[1])
    return w * h


def intersection(a: list[float], b: list[float]) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    return (x2 - x1) * (y2 - y1)


def iou(a: list[float], b: list[float]) -> float:
    inter = intersection(a, b)
    if inter <= 0.0:
        return 0.0
    union = area(a) + area(b) - inter
    return inter / union if union > 0 else 0.0


def center(bbox: list[float]) -> tuple[float, float]:
    return ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)


def scale_bbox(bbox: list[float], sx: float, sy: float) -> list[float]:
    return [bbox[0] * sx, bbox[1] * sy, bbox[2] * sx, bbox[3] * sy]
