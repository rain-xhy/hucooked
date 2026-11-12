"""Tensor state and the explicit host boundary for batched MixedKitchen."""

from dataclasses import asdict
from typing import NamedTuple

import torch

from .kitchen import DIRECTIONS, VERSION
from .model import Chef, Item, JointOperation, KitchenState, Order, Pot

EMPTY, DISH, INGREDIENT, SOUP = range(4)
ITEM_KIND, ITEM_ORDER, ITEM_CERTIFIED = range(3)
ITEM_NAMES = (None, "dish", "ingredient", "soup")
NORMAL, JOINT = range(2)
ORDER_KINDS = ("normal", "joint")
ACTIVE, COMPLETED, EXPIRED = range(3)
STATUS_NAMES = ("active", "completed", "expired")
ORDER_ID, CREATED, DUE, SYNC_DUE, COOK, SYNC, STATUS, STOCK, COMPLETED_AT, CERTIFIED = range(10)
ORDER_FIELDS = 10
POT_ORDER, REMAINING = range(2)

METRICS = (
    "normal_completed", "joint_completed", "normal_issued", "joint_issued",
    "batches_completed", "unfinished_at_close", "normal_missed", "joint_missed",
    "joint_starts", "joint_completions", "joint_aborts", "invalid_joint_presses",
    "wait_agent_steps", "wasted_ingredients", "collisions",
)
AGENT_METRICS = ("loads_by_agent", "serves_by_agent", "joint_ticks_by_agent")
METRIC_INDEX = {name: index for index, name in enumerate(METRICS)}
AGENT_METRIC_INDEX = {name: index for index, name in enumerate(AGENT_METRICS)}


class TorchKitchenState(NamedTuple):
    positions: torch.Tensor
    orientations: torch.Tensor
    held: torch.Tensor
    pots: torch.Tensor
    buffers: torch.Tensor
    orders: torch.Tensor
    progress: torch.Tensor
    pair: torch.Tensor
    donor: torch.Tensor
    tick: torch.Tensor
    batch: torch.Tensor
    score: torch.Tensor
    terminated: torch.Tensor
    truncated: torch.Tensor
    history: torch.Tensor
    history_count: torch.Tensor
    metrics: torch.Tensor
    agent_metrics: torch.Tensor
    latencies: torch.Tensor
    schedule_index: torch.Tensor


def encode_order(order):
    return [order.id, order.created_at, order.due_at, order.sync_due_at,
            order.cook_steps, order.sync_steps, STATUS_NAMES.index(order.status), order.stock,
            -1 if order.completed_at is None else order.completed_at, int(order.certified)]


def decode_order(row, kind):
    order_id, created, due, sync_due, cook, sync, status, stock, completed, certified = row
    return Order(order_id, kind, created, due, sync_due, cook, sync, STATUS_NAMES[status],
                 stock, None if completed < 0 else completed, bool(certified))


def encode_state(state, width, capacity):
    order_slots = {order.id: i for i, order in enumerate(state.orders)}

    def encode_item(item):
        if item is None:
            return [EMPTY, -1, 0]
        return [ITEM_NAMES.index(item.kind), order_slots.get(item.order_id, -1), int(item.certified)]

    history = [encode_order(Order(**order)) for order in state.history]
    latencies = state.metrics["latencies"]
    if len(history) > capacity or len(latencies) > capacity:
        raise ValueError("Snapshot exceeds the configured order history capacity")
    return TorchKitchenState(
        [x + y * width for x, y in (chef.position for chef in state.chefs)],
        [DIRECTIONS.index(chef.orientation) for chef in state.chefs],
        [encode_item(chef.held) for chef in state.chefs],
        [[order_slots.get(pot.order_id, -1), pot.remaining] for pot in state.pots],
        [encode_item(item) for item in state.buffers],
        [encode_order(order) for order in state.orders],
        state.joint.progress,
        [i in state.joint.pair for i in range(3)],
        -1 if state.joint.donor is None else state.joint.donor,
        state.tick, state.batch, state.score, state.terminated, state.truncated,
        history + [[0] * ORDER_FIELDS] * (capacity - len(history)), len(history),
        [state.metrics[name] for name in METRICS],
        [state.metrics[name] for name in AGENT_METRICS],
        latencies + [0] * (capacity - len(latencies)),
        0,
    )


def native_metrics(metrics, agent_metrics, latencies):
    result = dict(zip(METRICS, metrics))
    result.update(zip(AGENT_METRICS, agent_metrics))
    deliveries = result["normal_completed"] + result["joint_completed"]
    result["latencies"] = latencies[:deliveries]
    return result


def decode_state(state, lane, width):
    values = TorchKitchenState(*(tensor[lane].tolist() for tensor in state))
    orders = [decode_order(row, kind) for row, kind in zip(values.orders, ORDER_KINDS)]

    def order_id(slot):
        return None if slot < 0 else orders[slot].id

    def decode_item(row):
        kind, slot, certified = row
        return Item(ITEM_NAMES[kind], order_id(slot), bool(certified)) if kind else None

    chefs = [
        Chef((cell % width, cell // width), DIRECTIONS[orientation], decode_item(item))
        for cell, orientation, item in zip(values.positions, values.orientations, values.held)
    ]
    joint = JointOperation()
    if values.progress:
        joint = JointOperation(orders[JOINT].id, tuple(i for i, flag in enumerate(values.pair) if flag),
                               values.donor, values.progress)
    return KitchenState(
        chefs=chefs,
        pots=[Pot(order_id(slot), remaining) for slot, remaining in values.pots],
        buffers=[decode_item(item) for item in values.buffers],
        orders=orders, joint=joint, tick=values.tick, batch=values.batch, score=values.score,
        terminated=values.terminated, truncated=values.truncated,
        history=[asdict(decode_order(row, ORDER_KINDS[i % 2]))
                 for i, row in enumerate(values.history[:values.history_count])],
        metrics=native_metrics(values.metrics, values.agent_metrics, values.latencies),
    )


EVENT_FIELDS = {
    "joint_aborted": ("order_id", "progress_lost"),
    "joint_started": ("order_id", "pair"),
    "joint_progress": ("progress", "required", "pair"),
    "joint_completed": ("order_id", "donor", "receiver"),
    "joint_not_ready": ("agents",),
    "resource_conflict": ("agents", "position"),
    "ingredient_taken": ("agent", "order_id"),
    "ingredient_returned": ("agent", "order_id"),
    "dish_returned": ("agent",),
    "pot_loaded": ("agent", "order_id"),
    "soup_plated": ("agent", "order_id"),
    "delivery_rejected": ("agent", "order_id"),
    "delivered": ("agent", "kind", "order_id"),
    "collision": ("agents",),
    "order_expired": ("order_id", "kind"),
    "batch_completed": ("batch",),
}


class TorchKitchenInfo(NamedTuple):
    """Per-lane metrics and ordered event masks on the same device as the state."""

    tick: torch.Tensor
    batch: torch.Tensor
    score: torch.Tensor
    raw_reward: torch.Tensor
    shaping_reward: torch.Tensor
    metrics: torch.Tensor
    agent_metrics: torch.Tensor
    latencies: torch.Tensor
    events: tuple
    fingerprint: str
    width: int
    active: tuple

    def to_native(self, lane=0):
        events = []
        if self.events:
            masks = torch.stack([mask[lane] for _, mask, _ in self.events]).tolist()
            for (kind, _, tensors), present in zip(self.events, masks):
                if not present:
                    continue
                values = torch.stack([tensor[lane] for tensor in tensors]).tolist()
                event = {"type": kind}
                for field, value in zip(EVENT_FIELDS[kind], values):
                    if field == "agents":
                        agents = range(3) if kind == "collision" else self.active
                        value = [i for i in agents if value & (1 << i)]
                    elif field == "pair":
                        value = tuple(i for i in range(3) if value & (1 << i))
                    elif field == "position":
                        value = (value % self.width, value // self.width)
                    elif field == "kind":
                        value = ORDER_KINDS[value]
                    event[field] = value
                events.append(event)
        return {
            "version": VERSION, "fingerprint": self.fingerprint,
            "tick": self.tick[lane].item(), "batch": self.batch[lane].item(),
            "score": self.score[lane].item(), "raw_reward": self.raw_reward[lane].item(),
            "shaping_reward": self.shaping_reward[lane].item(), "events": events,
            "metrics": native_metrics(self.metrics[lane].tolist(), self.agent_metrics[lane].tolist(),
                                      self.latencies[lane].tolist()),
        }
