import unittest

from vln_core.config.models import Pose2D
from vln_core.eval.metrics import path_length
from vln_core.mission.compiler import compile_instruction
from vln_core.planning.follower import TrajectoryFollower
from vln_core.planning.scoring import build_trajectory
from vln_core.runtime.lifecycle import MissionTracker
from vln_core.safety.corridor import summarize_scan
from vln_core.world_model.semantic_map import SemanticMapStore
from vln_perception.plugins.mock_grounding import MockGroundingBackend


class PipelineChainTest(unittest.TestCase):
    def test_instruction_to_progressive_safe_execution(self):
        mission = compile_instruction("move to the tree and then head to the yellow finish area")
        backend = MockGroundingBackend()
        store = SemanticMapStore()
        store.bulk_observe(backend.observe_mission(mission))
        start_pose = Pose2D(x=0.0, y=0.0, yaw_rad=0.0)
        plan = build_trajectory(mission, store.snapshot(), start_pose=start_pose)
        follower = TrajectoryFollower(waypoint_tolerance_m=0.3)
        follower.set_plan(plan)
        tracker = MissionTracker()
        tracker.start(mission, started_at_sec=0.0)
        safety = summarize_scan([10.0] * 21, -1.57, 0.157)

        progress_events = []
        for index, waypoint in enumerate(plan.waypoints, start=1):
            events = follower.update_pose(Pose2D(x=waypoint.x, y=waypoint.y, z=waypoint.z, yaw_rad=waypoint.yaw_rad))
            progress = tracker.advance(
                active_step_index=index,
                phase=events[-1]["type"] if events else "tracking",
                detail=f"waypoint {index}",
                current_time_sec=float(index),
                mission_complete=any(event["type"] == "mission_complete" for event in events),
            )
            progress_events.extend(events)

        self.assertEqual(len(plan.waypoints), 2)
        self.assertFalse(safety.blocked)
        self.assertTrue(any(event["type"] == "mission_complete" for event in progress_events))
        self.assertTrue(progress.mission_complete)
        self.assertGreater(path_length(plan.waypoints, start_pose=start_pose), 0.0)


if __name__ == "__main__":
    unittest.main()
