import unittest

from vln_core.mission.compiler import compile_instruction, split_instruction


class MissionCompilerTest(unittest.TestCase):
    def test_split_instruction(self):
        self.assertEqual(
            split_instruction("move to the tree and then head to the yellow finish area"),
            ["move to the tree", "head to the yellow finish area"],
        )

    def test_compile_instruction_preserves_targets(self):
        mission = compile_instruction("turn right, go to the bench")
        self.assertEqual([step.action for step in mission.steps], ["right", "forward"])
        self.assertEqual(mission.steps[-1].target_label, "bench")


if __name__ == "__main__":
    unittest.main()

