from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass(frozen=True)
class MissionStep:
    step_index: int
    action: str
    target_label: str = ""
    desired_yaw_rad: float = 0.0
    desired_distance_m: float = 2.0
    terminal: bool = False


@dataclass(frozen=True)
class Mission:
    mission_id: str
    raw_instruction: str
    steps: List[MissionStep]


@dataclass(frozen=True)
class MissionProgress:
    mission_id: str
    active_step_index: int
    total_steps: int
    phase: str
    detail: str = ""
    started_at_sec: float = 0.0
    elapsed_sec: float = 0.0
    mission_complete: bool = False


@dataclass(frozen=True)
class Pose2D:
    x: float
    y: float
    yaw_rad: float = 0.0
    z: float = 1.5


@dataclass(frozen=True)
class Waypoint:
    x: float
    y: float
    z: float
    yaw_rad: float


@dataclass(frozen=True)
class SemanticObservation:
    label: str
    x: float
    y: float
    z: float = 0.0
    confidence: float = 0.5
    source: str = "mock"
    yaw_rad: float = 0.0


@dataclass(frozen=True)
class SemanticLandmark:
    label: str
    x: float
    y: float
    z: float = 0.0
    confidence: float = 0.5
    observation_count: int = 1
    state: str = "tracked"


@dataclass(frozen=True)
class TrajectoryPlan:
    mission_id: str
    frame_id: str
    waypoints: List[Waypoint]
    target_speed: float = 1.0
    planner_mode: str = "heuristic_rollout"


@dataclass(frozen=True)
class SafetySummary:
    blocked: bool
    clearance_m: float
    reason: str


@dataclass(frozen=True)
class EpisodeOutcome:
    run_id: str
    episode_name: str
    success: bool
    score: float
    summary_path: str
    detail: str
    metrics: Dict[str, float] = field(default_factory=dict)
    mission_id: Optional[str] = None
