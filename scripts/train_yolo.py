#!/usr/bin/env python
"""Sentetik dataset üzerinde bir YOLO dedektörü fine-tune et (ultralytics).

Önkoşul:
    pip install ultralytics
    python scripts/build_dataset.py --out dataset

Çalıştır (GPU önerilir; CPU'da yavaş ama çalışır):
    python scripts/train_yolo.py --data dataset/data.yaml --epochs 50

Çıktı: runs/detect/train/weights/best.pt
Sonraki adım: bu ağırlıkları bir `Prediction` adaptörüne sarıp
`hatscan.eval.run_eval(predict=...)` ile kural-tabanlı baseline'a karşı yarıştır.
"""
from __future__ import annotations

import argparse


def main() -> int:
    ap = argparse.ArgumentParser(description="Sentetik veride YOLO fine-tune")
    ap.add_argument("--data", default="dataset/data.yaml")
    ap.add_argument("--model", default="yolo11n-pose.pt", help="baslangic agirligi (pose); detect icin yolo11n.pt")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--imgsz", type=int, default=640)
    args = ap.parse_args()

    try:
        from ultralytics import YOLO
    except ImportError:
        raise SystemExit("ultralytics kurulu degil. Kur: pip install ultralytics")

    model = YOLO(args.model)
    model.train(data=args.data, epochs=args.epochs, imgsz=args.imgsz)
    print("[train] bitti -> runs/detect/train/weights/best.pt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
