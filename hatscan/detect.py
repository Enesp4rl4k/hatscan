"""Direk ve iletken tespiti (OpenCV — MVP)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

import cv2
import numpy as np

Point = Tuple[int, int]


@dataclass
class PoleCandidate:
    base: Point
    top: Point
    tilt_deg: float


@dataclass
class WireSegment:
    p1: Point
    p2: Point
    sag_px: float = 0.0


@dataclass
class DetectionResult:
    poles: List[PoleCandidate] = field(default_factory=list)
    wires: List[WireSegment] = field(default_factory=list)
    broken_wire_suspected: bool = False


def detect_pole(bgr: np.ndarray) -> List[PoleCandidate]:
    """Koyu dikey yogunluk bandi = direk."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    y0, y1 = int(h * 0.12), h - 40
    dark = gray[y0:y1] < 95
    col_scores = dark.sum(axis=0)
    if col_scores.max() < 30:
        return []
    peak_x = int(np.argmax(col_scores))
    xs = np.arange(max(0, peak_x - 30), min(w, peak_x + 31))
    local = col_scores[xs]
    keep = xs[local >= col_scores.max() * 0.45]
    if keep.size == 0:
        keep = np.array([peak_x])
    pts: List[Point] = []
    for x in keep:
        for y in range(y0, y1):
            if gray[y, x] < 95:
                pts.append((int(x), int(y)))
    if len(pts) < 40:
        return []
    arr = np.array(pts, dtype=np.float32)
    line = cv2.fitLine(arr, cv2.DIST_L2, 0, 0.01, 0.01).flatten()
    vx, vy = float(line[0]), float(line[1])
    if vy < 0:
        vx, vy = -vx, -vy
    t_vals = (arr[:, 0] - arr[:, 0].mean()) * vx + (arr[:, 1] - arr[:, 1].mean()) * vy
    base_pt = (int(arr[t_vals.argmax(), 0]), int(arr[t_vals.argmax(), 1]))
    top_pt = (int(arr[t_vals.argmin(), 0]), int(arr[t_vals.argmin(), 1]))
    if base_pt[1] < top_pt[1]:
        base_pt, top_pt = top_pt, base_pt
    shaft = arr[arr[:, 1] > top_pt[1] + (base_pt[1] - top_pt[1]) * 0.25]
    if len(shaft) >= 20:
        bi = int(shaft[:, 1].argmax())
        ti = int(shaft[:, 1].argmin())
        base_pt = (int(shaft[bi, 0]), int(shaft[bi, 1]))
        top_pt = (int(shaft[ti, 0]), int(shaft[ti, 1]))
    tilt = float(np.degrees(np.arctan2(top_pt[0] - base_pt[0], base_pt[1] - top_pt[1])))
    return [PoleCandidate(base_pt, top_pt, tilt)]


def detect_wires(bgr: np.ndarray) -> List[WireSegment]:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 130)
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180, threshold=45,
        minLineLength=80, maxLineGap=8,
    )
    wires: List[WireSegment] = []
    if lines is None:
        return wires
    for ln in lines:
        x1, y1, x2, y2 = ln[0]
        length = np.hypot(x2 - x1, y2 - y1)
        if length < 80:
            continue
        angle = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
        if angle > 25 and angle < 155:
            continue
        wires.append(WireSegment((x1, y1), (x2, y2)))
    return _merge_similar_wires(wires)


def _merge_similar_wires(wires: List[WireSegment], y_tol: int = 12) -> List[WireSegment]:
    if not wires:
        return wires
    wires = sorted(wires, key=lambda w: (w.p1[1] + w.p2[1]) / 2)
    merged: List[WireSegment] = []
    group = [wires[0]]
    for w in wires[1:]:
        gy = sum((g.p1[1] + g.p2[1]) / 2 for g in group) / len(group)
        wy = (w.p1[1] + w.p2[1]) / 2
        if abs(wy - gy) < y_tol:
            group.append(w)
        else:
            merged.append(_span_group(group))
            group = [w]
    merged.append(_span_group(group))
    return merged


def _span_group(group: List[WireSegment]) -> WireSegment:
    xs, ys = [], []
    for w in group:
        xs.extend([w.p1[0], w.p2[0]])
        ys.extend([w.p1[1], w.p2[1]])
    p1, p2 = (min(xs), int(np.mean(ys))), (max(xs), int(np.mean(ys)))
    return WireSegment(p1, p2, sag_px=float(abs(p2[1] - p1[1])))


def analyze(bgr: np.ndarray) -> DetectionResult:
    poles = detect_pole(bgr)
    wires = detect_wires(bgr)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    broken = _broken_wire_heuristic(wires) or _broken_wire_scan(gray, poles)
    return DetectionResult(poles=poles, wires=wires, broken_wire_suspected=broken)


def _broken_wire_heuristic(wires: List[WireSegment]) -> bool:
    """Ayni y seviyesinde iki parca arasinda buyuk bosluk."""
    if len(wires) < 2:
        return False
    for i, a in enumerate(wires):
        ay = (a.p1[1] + a.p2[1]) / 2
        ax1, ax2 = sorted([a.p1[0], a.p2[0]])
        for b in wires[i + 1:]:
            by = (b.p1[1] + b.p2[1]) / 2
            if abs(ay - by) > 25:
                continue
            bx1, bx2 = sorted([b.p1[0], b.p2[0]])
            gap = max(bx1 - ax2, ax1 - bx2)
            if gap > 45:
                return True
    return False


def _broken_wire_scan(gray: np.ndarray, poles: List[PoleCandidate]) -> bool:
    """Kopuk tel: direk ortada, solda uzun parca, sagda kisa parca."""
    h, _w = gray.shape
    _ = poles
    for y in range(int(h * 0.18), int(h * 0.42)):
        row = gray[y] < 80
        xs = np.where(row)[0]
        if xs.size < 3:
            continue
        runs: List[Tuple[int, int]] = []
        start = int(xs[0])
        for i in range(1, len(xs)):
            if xs[i] - xs[i - 1] > 1:
                runs.append((start, int(xs[i - 1])))
                start = int(xs[i])
        runs.append((start, int(xs[-1])))
        if len(runs) != 3:
            continue
        left, mid, right = runs
        lw, mw, rw = left[1] - left[0], mid[1] - mid[0], right[1] - right[0]
        if mw <= 25 and lw > 200 and rw < 80:
            return True
    return False
