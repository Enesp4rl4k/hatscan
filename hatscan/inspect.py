"""Anomali kuralları ve rapor."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import List

import cv2
import numpy as np

from .detect import DetectionResult, PoleCandidate, WireSegment

TILT_WARN_DEG = 8.0
SAG_WARN_PX = 25.0


@dataclass
class Finding:
    code: str
    severity: str
    message: str


@dataclass
class InspectionReport:
    ok: bool
    findings: List[Finding] = field(default_factory=list)
    pole_count: int = 0
    wire_count: int = 0

    def to_text(self) -> str:
        lines = ["HATSCAN DENETIM RAPORU", f"Sonuc: {'TAMAM' if self.ok else 'UYARI'}", ""]
        lines.append(f"Direk: {self.pole_count}  |  Iletken parcasi: {self.wire_count}")
        if self.findings:
            lines.append("")
            lines.append("Bulgular:")
            for f in self.findings:
                lines.append(f"  [{f.severity}] {f.code}: {f.message}")
        return "\n".join(lines)


def _safe_print(text: str) -> None:
    try:
        print(text)
    except UnicodeEncodeError:
        print(text.encode("ascii", errors="replace").decode("ascii"))


def inspect(det: DetectionResult) -> InspectionReport:
    findings: List[Finding] = []
    for p in det.poles:
        if abs(p.tilt_deg) > TILT_WARN_DEG:
            findings.append(Finding(
                "POLE_TILT", "HIGH",
                f"Direk egimi {p.tilt_deg:.1f}° (esik {TILT_WARN_DEG}°)"))
    for w in det.wires:
        if w.sag_px > SAG_WARN_PX:
            findings.append(Finding(
                "WIRE_SAG", "MEDIUM",
                f"Iletken sarkmasi ~{w.sag_px:.0f}px"))
    if det.broken_wire_suspected:
        findings.append(Finding(
            "WIRE_BREAK", "HIGH",
            "Iletkende kopukluk / bosluk suphesi"))
    if not det.poles:
        findings.append(Finding("NO_POLE", "LOW", "Direk tespit edilemedi"))
    ok = not any(f.severity == "HIGH" for f in findings)
    return InspectionReport(
        ok=ok,
        findings=findings,
        pole_count=len(det.poles),
        wire_count=len(det.wires),
    )


def draw_overlay(bgr: np.ndarray, det: DetectionResult, report: InspectionReport) -> np.ndarray:
    out = bgr.copy()
    for p in det.poles:
        color = (0, 0, 255) if abs(p.tilt_deg) > TILT_WARN_DEG else (0, 200, 0)
        cv2.line(out, p.base, p.top, color, 3)
        cv2.putText(out, f"{p.tilt_deg:.1f}deg", p.top,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    for w in det.wires:
        cv2.line(out, w.p1, w.p2, (255, 120, 0), 2)
    y = 24
    for f in report.findings[:6]:
        cv2.putText(out, f.message[:70], (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (0, 0, 255) if f.severity == "HIGH" else (0, 140, 255), 1)
        y += 18
    return out
