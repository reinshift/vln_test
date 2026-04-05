from typing import Dict, Iterable, List

from vln_core.config.models import SemanticLandmark, SemanticObservation


class SemanticMapStore:
    def __init__(self) -> None:
        self._landmarks: Dict[str, SemanticLandmark] = {}

    def observe(self, observation: SemanticObservation) -> SemanticLandmark:
        current = self._landmarks.get(observation.label)
        if current is None:
            updated = SemanticLandmark(
                label=observation.label,
                x=observation.x,
                y=observation.y,
                z=observation.z,
                confidence=observation.confidence,
                observation_count=1,
                state="tracked",
            )
        else:
            total = current.observation_count + 1
            updated = SemanticLandmark(
                label=observation.label,
                x=((current.x * current.observation_count) + observation.x) / total,
                y=((current.y * current.observation_count) + observation.y) / total,
                z=((current.z * current.observation_count) + observation.z) / total,
                confidence=max(current.confidence, observation.confidence),
                observation_count=total,
                state="tracked",
            )
        self._landmarks[observation.label] = updated
        return updated

    def bulk_observe(self, observations: Iterable[SemanticObservation]) -> List[SemanticLandmark]:
        return [self.observe(item) for item in observations]

    def snapshot(self) -> List[SemanticLandmark]:
        return [self._landmarks[key] for key in sorted(self._landmarks)]

    def reset(self) -> None:
        self._landmarks.clear()

