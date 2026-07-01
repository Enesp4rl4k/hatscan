"""Sentetik sahneden YOLO-format ground-truth etiket üretimi.

Çizimi biz kontrol ettiğimiz için her nesnenin (direk, tel) tam sınırlayıcı
kutusunu hesaplayabiliriz — sıfır maliyetli, kusursuz etiketli veri. Bu, gerçek
bir dedektörü (YOLO/RT-DETR) fine-tune etmenin önkoşulu olan veriyi üretir.

Sınıflar NESNE düzeyindedir (pole, wire). Arıza (tilt/sag/break) ayrı bir sınıf
değil, tespit edilen nesnenin geometrisinden çıkarılan bir değerlendirmedir —
hatscan'in mevcut detect→assess ayrımıyla aynı.
"""
from __future__ import annotations

from typing import List, Sequence, Tuple

from .synthetic import POLE_CAP_RADIUS, POLE_THICKNESS, Point, SceneGeometry, WIRE_THICKNESS

CLASSES = ["pole", "wire"]
CLASS_ID = {name: i for i, name in enumerate(CLASSES)}

# (class_id, x_center, y_center, width, height) — hepsi 0..1 normalize.
YoloBox = Tuple[int, float, float, float, float]


def _bbox(points: Sequence[Point], pad: float, width: int, height: int) -> Tuple[float, float, float, float]:
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    x0 = max(0.0, min(xs) - pad)
    x1 = min(float(width), max(xs) + pad)
    y0 = max(0.0, min(ys) - pad)
    y1 = min(float(height), max(ys) + pad)
    return (
        (x0 + x1) / 2 / width,
        (y0 + y1) / 2 / height,
        (x1 - x0) / width,
        (y1 - y0) / height,
    )


def yolo_labels(geom: SceneGeometry) -> List[YoloBox]:
    labels: List[YoloBox] = []

    pole_pad = POLE_THICKNESS / 2 + POLE_CAP_RADIUS
    xc, yc, w, h = _bbox([geom.pole_base, geom.pole_top], pole_pad, geom.width, geom.height)
    labels.append((CLASS_ID["pole"], xc, yc, w, h))

    xc, yc, w, h = _bbox(geom.wire_points, WIRE_THICKNESS, geom.width, geom.height)
    labels.append((CLASS_ID["wire"], xc, yc, w, h))

    return labels


def format_yolo(labels: List[YoloBox]) -> str:
    return "\n".join(
        f"{c} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}" for c, xc, yc, w, h in labels
    )


# --- Pose / keypoint etiketleri --------------------------------------------
# Tek sınıf "structure", 5 keypoint. tilt = açı(base, top), sag = orta noktanın
# sol-sağ kirişine göre çöküşü → model bunları doğrudan öğrenir (kutu vermez).
KEYPOINTS = ["pole_base", "pole_top", "wire_left", "wire_mid", "wire_right"]
# Yatay flip altında keypoint eşleşmesi (augmentation): sol <-> sağ.
FLIP_IDX = [0, 1, 4, 3, 2]


def pose_keypoints(geom: SceneGeometry) -> List[Point]:
    wire = geom.wire_points
    left, right = wire[0], wire[-1]
    if geom.broken:
        gap_a, gap_b = wire[1], wire[2]
        mid = ((gap_a[0] + gap_b[0]) // 2, (gap_a[1] + gap_b[1]) // 2)
    else:
        mid = wire[1]
    return [geom.pole_base, geom.pole_top, left, mid, right]


def pose_label_line(geom: SceneGeometry) -> str:
    """Ultralytics pose formatı: `cls xc yc w h (kx ky v)*5` — hepsi normalize."""
    kpts = pose_keypoints(geom)
    pad = POLE_THICKNESS / 2 + POLE_CAP_RADIUS
    xc, yc, w, h = _bbox(kpts, pad, geom.width, geom.height)
    parts = [f"0 {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}"]
    for x, y in kpts:
        # Kadraj dışına taşan uçları sınıra kıstır → geçerli [0,1] etiketi.
        nx = min(1.0, max(0.0, x / geom.width))
        ny = min(1.0, max(0.0, y / geom.height))
        parts.append(f"{nx:.6f} {ny:.6f} 2")
    return " ".join(parts)
