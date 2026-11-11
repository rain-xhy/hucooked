"""Serializable state and validated rules for MixedKitchen-v1."""

from dataclasses import asdict, dataclass, field
import math
from typing import Optional


@dataclass(frozen=True)
class KitchenConfig:
    horizon: int = 300
    max_batches: int = 8
    cook_steps: int = 4
    sync_steps: int = 4
    normal_deadline: int = 90
    joint_deadline: int = 100
    sync_deadline: int = 80
    deadline_jitter: int = 4
    duration_jitter: int = 1
    normal_reward: float = 20.0
    joint_reward: float = 30.0
    batch_bonus: float = 10.0
    expiry_penalty: float = -5.0
    shaping_scale: float = 0.0
    shaping_gamma: float = 0.99
    layout: str = "compact"
    shuffle_players: bool = True
    active_players: tuple = (0, 1, 2)
    reset_mode: str = "standard"

    def __post_init__(self):
        object.__setattr__(self, "active_players", tuple(self.active_players))
        for key in ("horizon", "max_batches", "cook_steps", "sync_steps",
                    "normal_deadline", "joint_deadline", "sync_deadline"):
            value = getattr(self, key)
            if type(value) is not int or value <= 0:
                raise ValueError(key + " must be a positive integer")
        for key in ("deadline_jitter", "duration_jitter"):
            if type(getattr(self, key)) is not int or getattr(self, key) < 0:
                raise ValueError(key + " must be a nonnegative integer")
        if self.duration_jitter >= min(self.cook_steps, self.sync_steps):
            raise ValueError("Duration jitter must be smaller than operation durations")
        if self.deadline_jitter >= min(self.normal_deadline, self.sync_deadline):
            raise ValueError("Deadline jitter is too large")
        if self.sync_deadline >= self.joint_deadline:
            raise ValueError("The handoff deadline must precede the delivery deadline")
        if not 0 < self.shaping_gamma <= 1 or self.shaping_scale < 0:
            raise ValueError("Invalid potential-shaping configuration")
        for key in ("normal_reward", "joint_reward", "batch_bonus", "expiry_penalty",
                    "shaping_scale", "shaping_gamma"):
            if not math.isfinite(getattr(self, key)):
                raise ValueError(key + " must be finite")
        if min(self.normal_reward, self.joint_reward) <= 0 or self.batch_bonus < 0 or self.expiry_penalty > 0:
            raise ValueError("Invalid task rewards or expiry penalty")
        if not self.active_players or len(set(self.active_players)) != len(self.active_players):
            raise ValueError("active_players must contain distinct player slots")
        if any(type(i) is not int or i not in range(3) for i in self.active_players):
            raise ValueError("Player slots must be integers 0, 1, or 2")
        if self.layout not in ("compact", "offset", "bottleneck"):
            raise ValueError("Unknown layout")
        if self.reset_mode not in ("standard", "kitchen_ready", "handoff_ready"):
            raise ValueError("Unknown reset distribution")
        if type(self.shuffle_players) is not bool:
            raise ValueError("shuffle_players must be a boolean")
        if self.reset_mode != "standard" and len(self.active_players) < 2:
            raise ValueError("Prepared training resets require at least two chefs")

    def to_dict(self):
        return asdict(self)


@dataclass
class Item:
    kind: str  # ingredient, dish, soup
    order_id: Optional[int] = None
    certified: bool = False


@dataclass
class Chef:
    position: tuple
    orientation: tuple = (0, -1)
    held: Optional[Item] = None


@dataclass
class Order:
    id: int
    kind: str  # normal (tomato) or joint (onion)
    created_at: int
    due_at: int
    sync_due_at: int
    cook_steps: int
    sync_steps: int
    status: str = "active"
    stock: int = 1
    completed_at: Optional[int] = None
    certified: bool = False


@dataclass
class Pot:
    order_id: Optional[int] = None
    remaining: int = 0


@dataclass
class JointOperation:
    order_id: Optional[int] = None
    pair: tuple = ()
    donor: Optional[int] = None
    progress: int = 0


@dataclass
class KitchenState:
    chefs: list
    pots: list
    buffers: list
    orders: list = field(default_factory=list)
    joint: JointOperation = field(default_factory=JointOperation)
    tick: int = 0
    batch: int = 0
    score: float = 0.0
    terminated: bool = False
    truncated: bool = False
    history: list = field(default_factory=list)
    metrics: dict = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, data):
        data = dict(data)
        data["chefs"] = [
            Chef(tuple(c["position"]), tuple(c["orientation"]),
                 Item(**c["held"]) if c["held"] else None)
            for c in data["chefs"]
        ]
        data["pots"] = [Pot(**p) for p in data["pots"]]
        data["buffers"] = [Item(**b) if b else None for b in data["buffers"]]
        data["orders"] = [Order(**o) for o in data["orders"]]
        joint = dict(data["joint"])
        joint["pair"] = tuple(joint["pair"])
        data["joint"] = JointOperation(**joint)
        return cls(**data)
