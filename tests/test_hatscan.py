"""Hatscan tests."""
import unittest

from hatscan import detect, inspect, synthetic


class DetectTests(unittest.TestCase):
    def test_finds_pole_on_synthetic(self) -> None:
        img = synthetic.make_scene(pole_tilt_deg=3.0, wire_sag_px=5)
        det = detect.analyze(img)
        self.assertGreaterEqual(len(det.poles), 1)

    def test_tilt_triggers_warning(self) -> None:
        img = synthetic.make_scene(pole_tilt_deg=15.0)
        rep = inspect.inspect(detect.analyze(img))
        codes = [f.code for f in rep.findings]
        self.assertIn("POLE_TILT", codes)
        self.assertFalse(rep.ok)

    def test_broken_wire_flag(self) -> None:
        img = synthetic.make_scene(broken_wire=True)
        det = detect.analyze(img)
        self.assertTrue(det.broken_wire_suspected)


if __name__ == "__main__":
    unittest.main()
