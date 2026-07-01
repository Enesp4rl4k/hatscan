"""Hatscan değerlendirme harness'i — sentetik ground-truth'a karşı tespit kalitesi.

`synthetic.make_scene` görüntüleri *bilinen* fiziksel parametrelerle üretir (gerçek
direk eğimi, tel sarkması, kopukluk). Bu yüzden tespit çıktısını yargılamaya gerek
yok — doğru cevap zaten elimizde. Harness dataset'i tarar, `detect → inspect`
hattını çalıştırır ve arıza tipi başına precision / recall / yanlış-alarm oranı
ile tilt MAE üretir.

Önemli: ground-truth arıza eşikleri (FAULT_*/OK_*) ürünün uyarı eşiğinden (inspect
içindeki TILT_WARN_DEG / SAG_WARN_PX) AYRI tutulur. Net "sağlam" ve net "arızalı"
bölgelerde örnekleyip etiketleri döngüsellikten arındırıyoruz; gri bölge dışarıda.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Set

from . import detect, inspect, synthetic

# Net arızalı / net sağlam bölge sınırları (ürün eşiğinden bağımsız).
FAULT_TILT_DEG = 12.0
OK_TILT_DEG = 4.0
FAULT_SAG_PX = 45
OK_SAG_PX = 10

FAULT_CODES = ["POLE_TILT", "WIRE_SAG", "WIRE_BREAK"]


@dataclass(frozen=True)
class Case:
    id: str
    tilt_deg: float
    sag_px: int
    broken: bool
    seed: int

    @property
    def truth(self) -> Dict[str, bool]:
        return {
            "POLE_TILT": abs(self.tilt_deg) >= FAULT_TILT_DEG,
            "WIRE_SAG": (not self.broken) and self.sag_px >= FAULT_SAG_PX,
            "WIRE_BREAK": self.broken,
        }


def default_dataset() -> List[Case]:
    """Dengeli, etiketleri temiz bir tarama: her seed için sağlam + 3 arıza tipi."""
    cases: List[Case] = []
    for seed in range(5):
        cases.append(Case(f"ok-{seed}", OK_TILT_DEG, OK_SAG_PX, False, seed))
        cases.append(Case(f"tilt-{seed}", FAULT_TILT_DEG + 5, OK_SAG_PX, False, seed))
        cases.append(Case(f"sag-{seed}", OK_TILT_DEG, FAULT_SAG_PX + 10, False, seed))
        cases.append(Case(f"break-{seed}", OK_TILT_DEG, 0, True, seed))
    return cases


@dataclass
class FaultMetrics:
    tp: int = 0
    fp: int = 0
    fn: int = 0
    tn: int = 0

    @property
    def precision(self) -> float:
        d = self.tp + self.fp
        return self.tp / d if d else 1.0

    @property
    def recall(self) -> float:
        d = self.tp + self.fn
        return self.tp / d if d else 1.0

    @property
    def false_alarm_rate(self) -> float:
        """FP / (FP + TN) — sağlam varlıkları boşuna işaretleme oranı."""
        d = self.fp + self.tn
        return self.fp / d if d else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0


@dataclass
class EvalReport:
    by_fault: Dict[str, FaultMetrics]
    tilt_mae: Optional[float]
    n_cases: int

    @property
    def macro_f1(self) -> float:
        return sum(m.f1 for m in self.by_fault.values()) / len(self.by_fault)


@dataclass
class Prediction:
    """Bir dedektörün tek görüntü için ürettiği çıktı.

    `codes`: tetiklenen arıza kodları (POLE_TILT / WIRE_SAG / WIRE_BREAK).
    `pole_tilt`: ölçülen direk eğimi (derece, mutlak) — yoksa None.
    """
    codes: Set[str]
    pole_tilt: Optional[float] = None


# Bir dedektör = görüntü → Prediction. Kural-tabanlı, YOLO, RT-DETR... hepsi uyar.
PredictFn = Callable[[object], Prediction]


def rule_based_predict(img) -> Prediction:
    """Mevcut OpenCV kural-tabanlı hat: detect.analyze → inspect.inspect."""
    det = detect.analyze(img)
    report = inspect.inspect(det)
    codes = {f.code for f in report.findings}
    tilt = max((abs(p.tilt_deg) for p in det.poles), default=None)
    return Prediction(codes=codes, pole_tilt=tilt)


def run_eval(
    cases: Optional[List[Case]] = None,
    predict: Optional[PredictFn] = None,
) -> EvalReport:
    """Dataset'i bir dedektöre karşı skorla.

    `predict` verilmezse kural-tabanlı hat kullanılır. Bir YOLO/RT-DETR adaptörü
    aynı imzayı (görüntü → Prediction) sağlayıp aynı karneye karşı yarışabilir —
    "hangi model daha iyi" sorusu böylece fikir değil, sayı olur.
    """
    cases = cases if cases is not None else default_dataset()
    predictor = predict or rule_based_predict
    metrics = {code: FaultMetrics() for code in FAULT_CODES}
    tilt_abs_errors: List[float] = []

    for case in cases:
        sag = 0 if case.broken else case.sag_px
        img = synthetic.make_scene(
            pole_tilt_deg=case.tilt_deg,
            wire_sag_px=sag,
            broken_wire=case.broken,
            seed=case.seed,
        )
        prediction = predictor(img)

        for code in FAULT_CODES:
            truth = case.truth[code]
            pred = code in prediction.codes
            m = metrics[code]
            if truth and pred:
                m.tp += 1
            elif truth and not pred:
                m.fn += 1
            elif pred:
                m.fp += 1
            else:
                m.tn += 1

        if prediction.pole_tilt is not None:
            tilt_abs_errors.append(abs(prediction.pole_tilt - abs(case.tilt_deg)))

    tilt_mae = sum(tilt_abs_errors) / len(tilt_abs_errors) if tilt_abs_errors else None
    return EvalReport(by_fault=metrics, tilt_mae=tilt_mae, n_cases=len(cases))


def format_report(r: EvalReport) -> str:
    lines = [
        "",
        "HATSCAN EVAL — sentetik ground-truth",
        f"Vaka sayisi: {r.n_cases}",
        "",
        f"{'Ariza':<12}{'Precision':>11}{'Recall':>9}{'F1':>7}{'YanlisAlarm':>13}",
        "-" * 52,
    ]
    for code in FAULT_CODES:
        m = r.by_fault[code]
        lines.append(
            f"{code:<12}{m.precision:>11.2f}{m.recall:>9.2f}{m.f1:>7.2f}{m.false_alarm_rate:>13.2f}"
        )
    lines.append("-" * 52)
    lines.append(f"{'MACRO F1':<12}{r.macro_f1:>27.2f}")
    mae = "n/a" if r.tilt_mae is None else f"{r.tilt_mae:.2f} deg"
    lines.append(f"Tilt MAE (olculen vs gercek aci): {mae}")
    return "\n".join(lines)
