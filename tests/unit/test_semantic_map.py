import unittest

from vln_core.config.models import SemanticObservation
from vln_core.world_model.semantic_map import SemanticMapStore


class SemanticMapTest(unittest.TestCase):
    def test_observation_fuses_positions(self):
        store = SemanticMapStore()
        store.observe(SemanticObservation(label="tree", x=4.0, y=6.0, confidence=0.7))
        item = store.observe(SemanticObservation(label="tree", x=6.0, y=6.0, confidence=0.8))
        self.assertEqual(item.observation_count, 2)
        self.assertAlmostEqual(item.x, 5.0)
        self.assertAlmostEqual(item.confidence, 0.8)


if __name__ == "__main__":
    unittest.main()

