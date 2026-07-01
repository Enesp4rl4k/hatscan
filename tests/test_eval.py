"""Eval harness tests."""
import math
import unittest

from hatscan import eval as hateval


class EvalHarnessTests(unittest.TestCase):
    def test_runs_and_reports_all_faults(self) -> None:
        report = hateval.run_eval()
        self.assertEqual(report.n_cases, len(hateval.default_dataset()))
        for code in hateval.FAULT_CODES:
            self.assertIn(code, report.by_fault)

    def test_metrics_are_in_range(self) -> None:
        report = hateval.run_eval()
        for m in report.by_fault.values():
            for value in (m.precision, m.recall, m.f1, m.false_alarm_rate):
                self.assertGreaterEqual(value, 0.0)
                self.assertLessEqual(value, 1.0)
        self.assertGreaterEqual(report.macro_f1, 0.0)
        self.assertLessEqual(report.macro_f1, 1.0)

    def test_broken_wire_is_detected(self) -> None:
        # Breakage is a deterministic flag in detect.analyze — recall should be perfect.
        report = hateval.run_eval()
        self.assertEqual(report.by_fault["WIRE_BREAK"].recall, 1.0)

    def test_clearly_ok_scenes_are_not_over_flagged(self) -> None:
        # Sağlam sahnelerde tilt yanlış-alarmı makul kalmalı (gri bölge yok).
        report = hateval.run_eval()
        self.assertLessEqual(report.by_fault["POLE_TILT"].false_alarm_rate, 0.5)

    def test_tilt_mae_is_finite(self) -> None:
        report = hateval.run_eval()
        self.assertIsNotNone(report.tilt_mae)
        assert report.tilt_mae is not None
        self.assertTrue(math.isfinite(report.tilt_mae))


if __name__ == "__main__":
    unittest.main()
