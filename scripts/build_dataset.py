#!/usr/bin/env python
"""Sentetik YOLO eğitim/validasyon dataseti üret (etiketli — bedava).

    python scripts/build_dataset.py --out dataset --per-combo 4

Çıktı:
    <out>/images/{train,val}/*.png
    <out>/labels/{train,val}/*.txt   (YOLO formatı)
    <out>/data.yaml                  (ultralytics ile eğitime hazır)
"""
from __future__ import annotations

import argparse
import os
import random
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import cv2  # noqa: E402

from hatscan import labels, synthetic  # noqa: E402

TILTS = [0, 6, 12, 18]
SAGS = [0, 20, 45]
BROKENS = [False, True]


def main() -> int:
    ap = argparse.ArgumentParser(description="Sentetik YOLO dataseti uret")
    ap.add_argument("--out", default="dataset")
    ap.add_argument("--per-combo", type=int, default=4, help="her parametre kombinasyonu icin gorsel")
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--task", choices=["pose", "detect"], default="pose",
                    help="pose = keypoint (tilt/sag icin); detect = bbox")
    args = ap.parse_args()

    combos = []
    idx = 0
    for tilt in TILTS:
        for sag in SAGS:
            for broken in BROKENS:
                for _ in range(args.per_combo):
                    combos.append((tilt, 0 if broken else sag, broken, idx))
                    idx += 1
    random.Random(0).shuffle(combos)
    n_val = int(len(combos) * args.val_frac)

    for split in ("train", "val"):
        os.makedirs(os.path.join(args.out, "images", split), exist_ok=True)
        os.makedirs(os.path.join(args.out, "labels", split), exist_ok=True)

    for i, (tilt, sag, broken, seed) in enumerate(combos):
        split = "val" if i < n_val else "train"
        img = synthetic.make_scene(pole_tilt_deg=tilt, wire_sag_px=sag, broken_wire=broken, seed=seed)
        geom = synthetic.scene_geometry(pole_tilt_deg=tilt, wire_sag_px=sag, broken_wire=broken)
        name = f"scene_{i:04d}"
        cv2.imwrite(os.path.join(args.out, "images", split, name + ".png"), img)
        text = (
            labels.pose_label_line(geom)
            if args.task == "pose"
            else labels.format_yolo(labels.yolo_labels(geom))
        )
        with open(os.path.join(args.out, "labels", split, name + ".txt"), "w", encoding="utf-8") as f:
            f.write(text)

    if args.task == "pose":
        data_yaml = (
            f"path: {os.path.abspath(args.out)}\n"
            "train: images/train\nval: images/val\n"
            "kpt_shape: [5, 3]\n"
            f"flip_idx: {labels.FLIP_IDX}\n"
            "names:\n  0: structure\n"
        )
    else:
        names = "\n".join(f"  {i}: {n}" for i, n in enumerate(labels.CLASSES))
        data_yaml = (
            f"path: {os.path.abspath(args.out)}\ntrain: images/train\nval: images/val\nnames:\n{names}\n"
        )
    with open(os.path.join(args.out, "data.yaml"), "w", encoding="utf-8") as f:
        f.write(data_yaml)

    model_hint = "yolo11n-pose.pt" if args.task == "pose" else "yolo11n.pt"
    print(f"[dataset] {len(combos)} gorsel ({args.task}) -> {args.out}  (train={len(combos) - n_val}, val={n_val})")
    print(f"[dataset] data.yaml hazir -> egit: python scripts/train_yolo.py --data {args.out}/data.yaml --model {model_hint}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
