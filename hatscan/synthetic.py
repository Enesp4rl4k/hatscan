"""Sentetik test görüntüsü: direk + iletkenler (+ ground-truth geometri).

`scene_geometry` çizim koordinatlarını döndürür; `make_scene` onları kullanarak
görüntüyü çizer. Aynı geometri `labels.py` tarafından YOLO etiketi üretmek için
kullanılır — yani her görüntünün kutusu *bilinen*, sıfır maliyetli etiketli veri.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np

Point = Tuple[int, int]

POLE_THICKNESS = 14
WIRE_THICKNESS = 3
POLE_CAP_RADIUS = 8


@dataclass(frozen=True)
class SceneGeometry:
    width: int
    height: int
    pole_base: Point
    pole_top: Point
    wire_points: List[Point]
    broken: bool


def scene_geometry(
    width: int = 800,
    height: int = 600,
    *,
    pole_tilt_deg: float = 0.0,
    wire_sag_px: int = 0,
    broken_wire: bool = False,
) -> SceneGeometry:
    """Sahnedeki direk ve tel koordinatlarını (çizimden bağımsız) hesapla."""
    px, py = width // 2, height - 80
    ph = 320
    angle = np.radians(pole_tilt_deg)
    top_x = int(px + ph * np.sin(angle))
    top_y = int(py - ph * np.cos(angle))

    span = width // 2 - 40
    y_attach = top_y + 20
    left = (top_x - span, y_attach)
    right = (top_x + span, y_attach)
    mid_x = (left[0] + right[0]) // 2
    mid_y = y_attach + wire_sag_px

    if broken_wire:
        wire_points = [left, (mid_x - 40, mid_y), (mid_x + 50, mid_y + 10), right]
    else:
        wire_points = [left, (mid_x, mid_y), right]

    return SceneGeometry(width, height, (px, py), (top_x, top_y), wire_points, broken_wire)


def make_scene(
    width: int = 800,
    height: int = 600,
    *,
    pole_tilt_deg: float = 0.0,
    wire_sag_px: int = 0,
    broken_wire: bool = False,
    seed: int = 0,
) -> np.ndarray:
    """BGR görüntü üret. wire_sag_px>0 → tel sarkması simülasyonu."""
    rng = np.random.default_rng(seed)
    img = np.full((height, width, 3), (200, 210, 220), dtype=np.uint8)
    # gökyüzü
    for y in range(height // 3):
        img[y, :] = (235, 180, 140)

    geom = scene_geometry(
        width, height,
        pole_tilt_deg=pole_tilt_deg, wire_sag_px=wire_sag_px, broken_wire=broken_wire,
    )

    cv2.line(img, geom.pole_base, geom.pole_top, (60, 55, 50), POLE_THICKNESS)
    cv2.circle(img, geom.pole_top, POLE_CAP_RADIUS, (40, 40, 45), -1)

    if broken_wire:
        left, gap_a, gap_b, right = geom.wire_points
        cv2.line(img, left, gap_a, (30, 30, 35), WIRE_THICKNESS)
        cv2.line(img, gap_b, right, (30, 30, 35), WIRE_THICKNESS)
    else:
        pts = np.array(geom.wire_points, np.int32)
        cv2.polylines(img, [pts], False, (25, 25, 30), WIRE_THICKNESS)

    # gürültü
    for _ in range(200):
        x, y = int(rng.integers(0, width)), int(rng.integers(0, height))
        img[y, x] = tuple(int(c + rng.integers(-15, 15)) for c in img[y, x])
    return img
