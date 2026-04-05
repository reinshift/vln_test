from typing import Optional

from vln_core.config.models import Mission, MissionProgress


class MissionTracker:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._mission: Optional[Mission] = None
        self._progress = MissionProgress(
            mission_id="",
            active_step_index=0,
            total_steps=0,
            phase="idle",
            detail="awaiting mission",
        )

    def start(self, mission: Mission, started_at_sec: float) -> MissionProgress:
        self._mission = mission
        self._progress = MissionProgress(
            mission_id=mission.mission_id,
            active_step_index=1 if mission.steps else 0,
            total_steps=len(mission.steps),
            phase="mission_active",
            detail=f"started {len(mission.steps)} steps",
            started_at_sec=started_at_sec,
            elapsed_sec=0.0,
            mission_complete=False,
        )
        return self._progress

    def advance(
        self,
        active_step_index: int,
        phase: str,
        detail: str,
        current_time_sec: float,
        mission_complete: bool = False,
    ) -> MissionProgress:
        elapsed_sec = 0.0
        if self._progress.started_at_sec > 0.0:
            elapsed_sec = max(0.0, current_time_sec - self._progress.started_at_sec)
        self._progress = MissionProgress(
            mission_id=self._progress.mission_id,
            active_step_index=active_step_index,
            total_steps=self._progress.total_steps,
            phase=phase,
            detail=detail,
            started_at_sec=self._progress.started_at_sec,
            elapsed_sec=elapsed_sec,
            mission_complete=mission_complete,
        )
        return self._progress

    def snapshot(self, current_time_sec: float) -> MissionProgress:
        return self.advance(
            active_step_index=self._progress.active_step_index,
            phase=self._progress.phase,
            detail=self._progress.detail,
            current_time_sec=current_time_sec,
            mission_complete=self._progress.mission_complete,
        )

