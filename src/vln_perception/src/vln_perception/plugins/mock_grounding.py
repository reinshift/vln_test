from vln_core.config.models import SemanticObservation
from vln_perception.plugins.base import GroundingBackend


class MockGroundingBackend(GroundingBackend):
    def __init__(self, landmarks=None):
        self.landmarks = landmarks or {
            "bench": (8.0, 2.0, 0.0),
            "tree": (4.0, 6.0, 0.0),
            "yellow finish area": (12.0, 4.0, 0.0),
            "finish area": (12.0, 4.0, 0.0),
            "cones": (2.0, 1.5, 0.0),
            "cone gate": (2.0, 1.5, 0.0),
            "marker board": (10.0, -2.0, 0.0),
        }
        self.debug_box_presets = {
            "tree": (0.40, 0.14, 0.66, 0.86),
            "bench": (0.56, 0.58, 0.88, 0.84),
            "yellow finish area": (0.30, 0.70, 0.78, 0.94),
            "finish area": (0.30, 0.70, 0.78, 0.94),
            "marker board": (0.64, 0.26, 0.84, 0.68),
            "cones": (0.14, 0.62, 0.42, 0.88),
            "cone gate": (0.14, 0.62, 0.42, 0.88),
        }

    def observe_mission(self, mission):
        observations = []
        for step in mission.steps:
            if not step.target_label:
                continue
            x, y, z = self.landmarks.get(step.target_label, (float(step.step_index) * 2.0, 0.0, 0.0))
            observations.append(
                SemanticObservation(
                    label=step.target_label,
                    x=x,
                    y=y,
                    z=z,
                    confidence=0.95,
                    source="mock_grounding",
                )
            )
        return observations

    def debug_boxes_for_mission(self, mission):
        boxes = []
        seen = set()
        fallback_slots = (
            (0.18, 0.20, 0.42, 0.80),
            (0.40, 0.18, 0.66, 0.78),
            (0.62, 0.22, 0.88, 0.74),
        )
        for step in mission.steps:
            label = step.target_label.strip()
            if not label or label in seen:
                continue
            seen.add(label)
            xmin, ymin, xmax, ymax = self.debug_box_presets.get(label, fallback_slots[len(boxes) % len(fallback_slots)])
            boxes.append(
                {
                    "label": label,
                    "confidence": 0.95,
                    "xmin": xmin,
                    "ymin": ymin,
                    "xmax": xmax,
                    "ymax": ymax,
                    "source": "mock_grounding",
                }
            )
        return boxes
