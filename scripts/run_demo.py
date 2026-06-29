#!/usr/bin/env python
"""Hatscan demo — sentetik veya dosyadan görüntü denetle."""
from __future__ import annotations

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import cv2  # noqa: E402

from hatscan import detect, inspect, synthetic  # noqa: E402
from hatscan.inspect import _safe_print  # noqa: E402

OUTPUT = os.path.join(ROOT, "outputs")


def main() -> int:
    p = argparse.ArgumentParser(description="Hatscan direk/hat denetimi")
    p.add_argument("--image", help="denetlenecek foto (yoksa sentetik)")
    p.add_argument("--tilt", type=float, default=12.0, help="sentetik: direk egimi derece")
    p.add_argument("--sag", type=int, default=35, help="sentetik: tel sarkmasi px")
    p.add_argument("--broken", action="store_true", help="sentetik: kopuk tel")
    args = p.parse_args()

    os.makedirs(OUTPUT, exist_ok=True)

    if args.image and os.path.isfile(args.image):
        bgr = cv2.imread(args.image)
        if bgr is None:
            print(f"[!] Okunamadi: {args.image}")
            return 1
        print(f"[*] Gorsel: {args.image}")
    else:
        sag = 0 if args.broken else args.sag
        bgr = synthetic.make_scene(
            pole_tilt_deg=args.tilt, wire_sag_px=sag,
            broken_wire=args.broken)
        print("[*] Sentetik test gorseli")

    det = detect.analyze(bgr)
    report = inspect.inspect(det)
    overlay = inspect.draw_overlay(bgr, det, report)

    cv2.imwrite(os.path.join(OUTPUT, "input.png"), bgr)
    cv2.imwrite(os.path.join(OUTPUT, "report.png"), overlay)
    with open(os.path.join(OUTPUT, "report.json"), "w", encoding="utf-8") as f:
        json.dump({
            "ok": report.ok,
            "pole_count": report.pole_count,
            "wire_count": report.wire_count,
            "findings": [f.__dict__ for f in report.findings],
        }, f, indent=2, ensure_ascii=False)
    with open(os.path.join(OUTPUT, "report.txt"), "w", encoding="utf-8") as f:
        f.write(report.to_text())

    _safe_print(report.to_text())
    print(f"\n[*] Cikti: {OUTPUT}/")
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
