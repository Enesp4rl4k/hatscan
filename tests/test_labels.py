"""YOLO etiket üretimi testleri."""
import unittest

from hatscan import labels, synthetic


class LabelTests(unittest.TestCase):
    def test_two_object_classes_normalized(self) -> None:
        geom = synthetic.scene_geometry(pole_tilt_deg=10.0, wire_sag_px=30)
        lbls = labels.yolo_labels(geom)
        self.assertEqual(sorted(box[0] for box in lbls), [0, 1])
        for _cls, xc, yc, w, h in lbls:
            for value in (xc, yc, w, h):
                self.assertGreaterEqual(value, 0.0)
                self.assertLessEqual(value, 1.0)
            self.assertGreater(w, 0.0)
            self.assertGreater(h, 0.0)

    def test_pole_bbox_contains_pole_endpoints(self) -> None:
        w_img, h_img = 800, 600
        geom = synthetic.scene_geometry(w_img, h_img, pole_tilt_deg=0.0, wire_sag_px=0)
        pole = next(box for box in labels.yolo_labels(geom) if box[0] == labels.CLASS_ID["pole"])
        _cls, xc, yc, w, h = pole
        x0, x1 = (xc - w / 2) * w_img, (xc + w / 2) * w_img
        y0, y1 = (yc - h / 2) * h_img, (yc + h / 2) * h_img
        for px, py in (geom.pole_base, geom.pole_top):
            self.assertTrue(x0 - 1 <= px <= x1 + 1)
            self.assertTrue(y0 - 1 <= py <= y1 + 1)

    def test_format_round_trips_to_five_fields(self) -> None:
        geom = synthetic.scene_geometry()
        text = labels.format_yolo(labels.yolo_labels(geom))
        rows = [r for r in text.strip().splitlines() if r]
        self.assertEqual(len(rows), 2)
        for row in rows:
            self.assertEqual(len(row.split()), 5)


if __name__ == "__main__":
    unittest.main()
