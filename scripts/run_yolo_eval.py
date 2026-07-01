#!/usr/bin/env python
"""Kural-tabanlı baseline vs eğitilmiş YOLO-pose → yan yana karne.

    python scripts/run_yolo_eval.py --weights runs/pose/train/weights/best.pt

Aynı sentetik ground-truth dataset'ine karşı iki dedektörü skorlar. WIRE_SAG
satırına bak: kural-tabanlı recall 0.00 → YOLO bunu kapatmalı.
"""
from __future__ import annotations

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from hatscan import eval as hateval  # noqa: E402
from hatscan.inspect import _safe_print  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="Baseline vs YOLO-pose karne")
    ap.add_argument("--weights", required=True, help="egitilmis pose .pt")
    args = ap.parse_args()

    baseline = hateval.run_eval()
    _safe_print("=== BASELINE (kural-tabanli) ===")
    _safe_print(hateval.format_report(baseline))

    from hatscan.yolo_pose import make_pose_predict

    yolo = hateval.run_eval(predict=make_pose_predict(args.weights))
    _safe_print("\n=== YOLO-pose ===")
    _safe_print(hateval.format_report(yolo))

    _safe_print(
        f"\nMacro F1: kural-tabanli {baseline.macro_f1:.2f}  ->  YOLO {yolo.macro_f1:.2f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
