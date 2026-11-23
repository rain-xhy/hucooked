from dataclasses import dataclass, field
import math


@dataclass(frozen=True)
class SharedKitchenConfig:
    horizon: int = 300
    normal_orders: int = 4
    joint_orders: int = 4
    active_limit: int = 2
    cook_steps: int = 4
    chop_steps: int = 2
    normal_reward: float = 20.0
    joint_reward: float = 20.0
    shuffle_players: bool = True
    active_players: tuple = (0, 1, 2)

    def __post_init__(self):
        for name in ("horizon", "active_limit", "cook_steps", "chop_steps"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(name + " must be a positive integer")
        for name in ("normal_orders", "joint_orders"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(name + " must be a nonnegative integer")
        if not 1 <= self.normal_orders + self.joint_orders <= 64:
            raise ValueError("The episode must have between 1 and 64 orders")
        for name in ("normal_reward", "joint_reward"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(name + " must be finite and nonnegative")
        if type(self.shuffle_players) is not bool:
            raise ValueError("shuffle_players must be boolean")
        players = tuple(self.active_players)
        if not players or any(type(i) is not int or i not in (0, 1, 2) for i in players) or len(set(players)) != len(players):
            raise ValueError("active_players must contain unique player IDs from 0, 1, 2")
        object.__setattr__(self, "active_players", tuple(sorted(players)))


@dataclass(frozen=True)
class Item:
    kind: str
    units: tuple
    order_id: int = -1
    contributors: tuple = ()
    prepared_by: tuple = ()


@dataclass
class Chef:
    position: tuple
    orientation: tuple = (0, -1)
    held: Item = None


@dataclass
class Cooker:
    item: Item = None
    remaining: int = 0


@dataclass
class Cutting:
    item: Item = None
    progress: int = 0


@dataclass
class Assembly:
    main: Item = None
    side: Item = None
    plate: Item = None


@dataclass
class Ticket:
    id: int
    kind: str
    status: str = "queued"
    released_at: int = -1
    started_at: int = -1
    cooked_at: int = -1
    unloaded_at: int = -1
    assembled_at: int = -1
    completed_at: int = -1
    main_loaded_by: tuple = ()
    unloaded_by: tuple = ()
    side_prepared_by: tuple = ()
    plate_supplied_by: tuple = ()
    assembled_by: tuple = ()
    served_by: tuple = ()
    contributors: tuple = ()
    delivered_units: tuple = ()


@dataclass
class KitchenState:
    chefs: list
    orders: list
    cooker: Cooker = field(default_factory=Cooker)
    cutting: Cutting = field(default_factory=Cutting)
    assembly: Assembly = field(default_factory=Assembly)
    buffer: Item = None
    tick: int = 0
    score: float = 0.0
    unit_sources: list = field(default_factory=list)
    returned_units: list = field(default_factory=list)
    wasted_units: list = field(default_factory=list)
    metrics: dict = field(default_factory=lambda: {
        "normal_completed": 0,
        "joint_completed": 0,
        "joint_ready_steps": 0,
        "joint_attempt_steps": 0,
        "joint_successes": 0,
        "unmatched_attempt_steps": 0,
        "resource_conflicts": 0,
        "movement_conflicts": 0,
    })
