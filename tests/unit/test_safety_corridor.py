import unittest

from vln_core.safety.corridor import summarize_scan


class SafetyCorridorTest(unittest.TestCase):
    def test_clear_scan(self):
        summary = summarize_scan([10.0] * 21, -1.57, 0.157, stop_distance_m=1.0)
        self.assertFalse(summary.blocked)
        self.assertEqual(summary.reason, "clear")

    def test_blocked_scan(self):
        ranges = [10.0] * 21
        ranges[10] = 0.8
        summary = summarize_scan(ranges, -1.57, 0.157, stop_distance_m=1.0)
        self.assertTrue(summary.blocked)
        self.assertEqual(summary.reason, "front_corridor_blocked")


if __name__ == "__main__":
    unittest.main()

