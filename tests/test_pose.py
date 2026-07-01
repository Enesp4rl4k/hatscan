"""Pose etiketi + keypoint→arıza geometrisi testleri (ultralytics gerekmez)."""
import math
import unittest

from hatscan import labels, synthetic, yolo_pose


class PoseLabelTests(unittest.TestCase):
    def test_pose_line_shape_and_visibility(self) -> None:
        geom = synthetic.scene_geometry(pole_tilt_deg=10.0, wire_sag_px=40)
        parts = labels.pose_label_line(geom).split()
        self.assertEqual(len(parts), 20)  # cls + 4 bbox + 5*(x,y,v)
        self.assertEqual(parts[0], "0")
        # bbox + keypoint koordinatları [0,1] aralığında olmalı (kadraj dışı kıstırılır)
        coord_idx = [1, 2, 3, 4, 5, 6, 8, 9, 11, 12, 14, 15, 17, 18]
        for i in coord_idx:
            self.assertGreaterEqual(float(parts[i]), 0.0)
            self.assertLessEqual(float(parts[i]), 1.0)
        for vis_idx in (7, 10, 13, 16, 19):
            self.assertEqual(parts[vis_idx], "2")

    def test_tilted_pole_keypoints_stay_in_frame(self) -> None:
        # Eğik direkte tel ucu kadrajı aşsa bile etiket [0,1]'de kalmalı.
        geom = synthetic.scene_geometry(pole_tilt_deg=18.0, wire_sag_px=0)
        parts = labels.pose_label_line(geom).split()
        coord_idx = [5, 6, 8, 9, 11, 12, 14, 15, 17, 18]
        for i in coord_idx:
            self.assertLessEqual(float(parts[i]), 1.0)
            self.assertGreaterEqual(float(parts[i]), 0.0)


class PoseGeometryTests(unittest.TestCase):
    def test_vertical_pole_has_zero_tilt(self) -> None:
        self.assertAlmostEqual(yolo_pose.tilt_from_keypoints((100, 500), (100, 100)), 0.0, places=5)

    def test_tilt_matches_atan2(self) -> None:
        tilt = yolo_pose.tilt_from_keypoints((100, 500), (150, 100))
        self.assertAlmostEqual(tilt, abs(math.degrees(math.atan2(50, 400))), places=5)

    def test_sag_is_chord_dip(self) -> None:
        sag = yolo_pose.sag_from_keypoints((0, 100), (100, 160), (200, 100))
        self.assertAlmostEqual(sag, 60.0, places=5)

    def test_codes_flag_tilt_and_sag(self) -> None:
        kpts = [(100, 500), (200, 100), (0, 100), (100, 200), (400, 100)]
        codes, tilt = yolo_pose.codes_from_keypoints(kpts)
        self.assertIn("POLE_TILT", codes)
        self.assertIn("WIRE_SAG", codes)
        self.assertGreater(tilt, yolo_pose.TILT_WARN_DEG)

    def test_codes_clean_when_straight(self) -> None:
        kpts = [(100, 500), (100, 100), (0, 100), (100, 105), (200, 100)]
        codes, _tilt = yolo_pose.codes_from_keypoints(kpts)
        self.assertNotIn("POLE_TILT", codes)
        self.assertNotIn("WIRE_SAG", codes)


if __name__ == "__main__":
    unittest.main()
