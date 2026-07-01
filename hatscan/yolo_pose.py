"""YOLO-pose adaptörü → hatscan.eval.Prediction.

Eğitilmiş bir keypoint modeli tek bir görüntü için 5 nokta üretir
(base, top, wire_left, wire_mid, wire_right). Buradan:
  • tilt = açı(base → top)            → POLE_TILT
  • sag  = orta noktanın sol-sağ kirişinin altına çöküşü → WIRE_SAG
Break ise keypoint'lerden çıkmaz (kopuk telin de uçları vardır), bu yüzden
mevcut klasik tespiti (zaten recall=1.0) kullanırız. Yani: algı modelden,
break klasikten — dürüst hibrit.

`make_pose_predict(weights)` → `eval.run_eval(predict=...)` imzasına uyan bir
fonksiyon döndürür; böylece YOLO, kural-tabanlı baseline'la aynı karneye girer.
"""
from __future__ import annotations

import math
from typing import Callable, Optional, Sequence

from . import detect, inspect
from .eval import Prediction

TILT_WARN_DEG = inspect.TILT_WARN_DEG
SAG_WARN_PX = inspect.SAG_WARN_PX

Pt = Sequence[float]


def tilt_from_keypoints(base: Pt, top: Pt) -> float:
    """Dikeyden sapma açısı (derece, mutlak) — inspect ile aynı tanım."""
    return abs(math.degrees(math.atan2(top[0] - base[0], base[1] - top[1])))


def sag_from_keypoints(left: Pt, mid: Pt, right: Pt) -> float:
    """Orta noktanın sol-sağ kirişinin altına dikey çöküşü (px)."""
    span = right[0] - left[0]
    if abs(span) < 1e-6:
        return 0.0
    t = (mid[0] - left[0]) / span
    chord_y = left[1] + t * (right[1] - left[1])
    return float(mid[1] - chord_y)


def codes_from_keypoints(kpts) -> tuple[set, Optional[float]]:
    base, top, left, mid, right = kpts[0], kpts[1], kpts[2], kpts[3], kpts[4]
    codes = set()
    tilt = tilt_from_keypoints(base, top)
    if tilt > TILT_WARN_DEG:
        codes.add("POLE_TILT")
    if sag_from_keypoints(left, mid, right) > SAG_WARN_PX:
        codes.add("WIRE_SAG")
    return codes, tilt


def make_pose_predict(weights: str) -> Callable[[object], Prediction]:
    """Eğitilmiş .pt ağırlığını yükleyip bir predict fonksiyonu döndür."""
    try:
        from ultralytics import YOLO
    except ImportError as exc:  # pragma: no cover - sadece çalışma zamanı
        raise SystemExit("ultralytics kurulu degil: pip install ultralytics") from exc

    model = YOLO(weights)

    def predict(img) -> Prediction:
        result = model(img, verbose=False)[0]
        codes: set = set()
        tilt: Optional[float] = None

        kp = getattr(result, "keypoints", None)
        if kp is not None and kp.xy is not None and len(kp.xy) > 0:
            points = kp.xy[0].cpu().numpy()
            if len(points) >= 5:
                codes, tilt = codes_from_keypoints(points)

        # Break: klasik tespit (robust, recall=1.0) ile birleştir.
        if detect.analyze(img).broken_wire_suspected:
            codes.add("WIRE_BREAK")

        return Prediction(codes=codes, pole_tilt=tilt)

    return predict
