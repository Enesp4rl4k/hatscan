#!/usr/bin/env python
"""Hatscan eval — sentetik ground-truth'a karşı tespit kalitesini skorla.

    python scripts/run_eval.py

Çıktı: arıza tipi başına precision / recall / F1 / yanlış-alarm + tilt MAE.
"""
from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from hatscan import eval as hateval  # noqa: E402
from hatscan.inspect import _safe_print  # noqa: E402


def main() -> int:
    report = hateval.run_eval()
    _safe_print(hateval.format_report(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
