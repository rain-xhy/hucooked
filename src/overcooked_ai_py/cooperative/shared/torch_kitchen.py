import numpy as np
import torch

from .kitchen import DIRECTIONS, INTERACT, STAY, ITEM_KINDS
from .layouts import ASSEMBLY_PORTS, COOKER_PORTS, FLOOR, LAYOUT, SPAWNS, STATIONS
from .model import SharedKitchenConfig


EMPTY, RAW_MAIN, RAW_SIDE, CHOPPED_SIDE, COOKED_MAIN, PLATE, MEAL = range(len(ITEM_KINDS))
ITEM_STRIDE = len(ITEM_KINDS)
QUEUED, ACTIVE, STARTED, SERVED = range(4)


def cell(position):
    x, y = position
    return y * len(LAYOUT[0]) + x


class TorchSharedKitchen:
    def __init__(self, config=None, *, device="cuda", num_envs=1):
        self.config = SharedKitchenConfig() if config is None else config
        if not isinstance(self.config, SharedKitchenConfig):
            raise TypeError("config must be SharedKitchenConfig")
        if type(num_envs) is not int or num_envs < 1:
            raise ValueError("num_envs must be a positive integer")
        self.num_envs = num_envs
        self.device = torch.empty(0, device=device).device
        config = self.config
        self.active = self.tensor([i in config.active_players for i in range(3)])
        self.ids = self.tensor(range(config.normal_orders + config.joint_orders))
        kinds = [kind for i in range(max(config.normal_orders, config.joint_orders))
                 for kind, count in enumerate((config.normal_orders, config.joint_orders)) if i < count]
        self.joint = self.tensor(kinds).bool()
        self.tiles = self.tensor([[tile == candidate for candidate in "X MLKGABDS"]
                                  for row in LAYOUT for tile in row]).flatten()
        self.ghost = sum(map(len, LAYOUT))
        self.floor = self.tensor([tile == " " for row in LAYOUT for tile in row] + [False])
        self.offset = self.tensor([dy * len(LAYOUT[0]) + dx for dx, dy in DIRECTIONS])
        self.spawn = [cell(pos) for pos in SPAWNS]
        self.stations = {k: cell(v) for k, v in STATIONS.items()}
        self.cooker_ports = self.tensor([cell(v) for v in COOKER_PORTS])
        self.assembly_ports = self.tensor([cell(v) for v in ASSEMBLY_PORTS])
        self.eyes = {n: torch.eye(n, device=self.device) for n in
                     {3, 4, 2, self.ghost + 1, ITEM_STRIDE, len(kinds) + 1}}
        self.tick = self.zeros()
        self.positions = self.zeros(3)
        self.orientation = self.zeros(3)
        self.held = self.zeros(3)
        self.cooker = self.zeros()
        self.cutting = self.zeros()
        self.main = self.zeros()
        self.side = self.zeros()
        self.plate = self.zeros()
        self.buffer = self.zeros()
        self.remaining = self.zeros()
        self.progress = self.zeros()
        self.status = self.zeros(len(kinds))
        self.released = self.zeros(len(kinds))
        self.started = self.zeros(len(kinds))
        self.rng = [np.random.default_rng(i) for i in range(num_envs)]
        self.reset(seed=0)

    def tensor(self, value):
        return torch.as_tensor(value, device=self.device)

    def zeros(self, *shape):
        return torch.zeros((self.num_envs, *shape), dtype=torch.long, device=self.device)

    def reset(self, seed=None, mask=None):
        lanes = list(range(self.num_envs))
        if mask is not None:
            mask = self.tensor(mask)
            if mask.shape != (self.num_envs,) or mask.dtype != torch.bool:
                raise ValueError("Expected one boolean reset flag per kitchen")
            lanes = mask.nonzero().flatten().tolist()
        if lanes:
            indices = self.tensor(lanes)
            positions = []
            for i in lanes:
                if seed is not None:
                    self.rng[i] = np.random.default_rng(seed + i)
                permutation = self.rng[i].permutation(3) if self.config.shuffle_players else range(3)
                positions.append([self.spawn[j] for j in permutation])
            self.positions[indices] = torch.where(self.active, self.tensor(positions), self.ghost)
            for field in (self.tick, self.orientation, self.held, self.cooker, self.cutting,
                          self.main, self.side, self.plate, self.buffer, self.remaining,
                          self.progress, self.released, self.started):
                field[indices] = 0
            self.status[indices] = (self.ids < self.config.active_limit).long()
        return self.observations(), {}

    def _item_features(self, items):
        return torch.cat((self.eyes[ITEM_STRIDE][items % ITEM_STRIDE],
                          self.eyes[len(self.ids) + 1][items // ITEM_STRIDE]), -1)

    def centralized_state(self):
        batch = self.num_envs
        horizon = self.config.horizon
        chefs = torch.cat((self.active.expand(batch, 3).unsqueeze(-1),
                           self.eyes[self.ghost + 1][self.positions][..., :self.ghost],
                           self.eyes[4][self.orientation], self._item_features(self.held)), -1)
        stations = torch.stack((self.cooker, self.cutting, self.main, self.side, self.plate, self.buffer), 1)
        orders = torch.cat((self.eyes[4][self.status], self.eyes[2][self.joint.long()].expand(batch, -1, -1),
                            self.released.unsqueeze(-1) / horizon, self.started.unsqueeze(-1) / horizon), -1)
        return torch.cat((self.tick[:, None] / horizon, self.tiles.expand(batch, -1), chefs.flatten(1),
                          self._item_features(stations).flatten(1), self.remaining[:, None] / self.config.cook_steps,
                          self.progress[:, None] / self.config.chop_steps, orders.flatten(1)), 1)

    def observations(self):
        common = self.centralized_state()[:, None].expand(-1, 3, -1)
        return torch.cat((common, self.eyes[3].expand(self.num_envs, -1, -1)), -1)

    def _move(self, actions):
        moving = actions < STAY
        self.orientation = torch.where(moving, actions, self.orientation)
        destination = (self.positions + self.offset[self.orientation]).clamp(0, self.ghost)
        desired = torch.where(moving & self.floor[destination], destination, self.positions)
        pairs = self.active[:, None] & self.active[None, :] & ~self.eyes[3].bool()
        follows = (desired[:, :, None] == self.positions[:, None, :]) & pairs
        blocked = (((desired[:, :, None] == desired[:, None, :]) & pairs) |
                   (follows & follows.transpose(1, 2))).any(2)
        for _ in range(2):
            blocked |= (follows & blocked[:, None, :]).any(2)
        self.positions = torch.where(blocked, self.positions, desired)

    def step(self, actions):
        actions = self.tensor(actions)
        if actions.shape != (self.num_envs, 3) or actions.dtype not in (
                torch.int8, torch.uint8, torch.int16, torch.int32, torch.int64):
            raise ValueError("Expected an integer action tensor of shape [num_envs, 3]")
        if ((actions < 0) | (actions > INTERACT) | (~self.active & (actions != STAY))).any().item():
            raise ValueError("Invalid action or inactive chef action")
        live = (self.tick < self.config.horizon) & (self.status != SERVED).any(1)
        actions = torch.where(live[:, None], actions.long(), STAY)
        target = torch.where(actions == INTERACT, self.positions + self.offset[self.orientation], -1)
        old = self.held
        kind = old % ITEM_STRIDE
        empty = kind == EMPTY
        held = old.clone()

        def at(station):
            return target == self.stations[station]

        def unique(request):
            return request & (request.sum(1, keepdim=True) == 1)

        def deposit(slot, request):
            return torch.where(request.any(1), torch.where(request, old, 0).sum(1), slot)

        def transfer(slot, put, take):
            nonlocal held
            held = torch.where(put, EMPTY, torch.where(take, slot[:, None], held))
            return torch.where(take.any(1), EMPTY, deposit(slot, put))

        for station, item in (("M", RAW_MAIN), ("L", RAW_SIDE), ("D", PLATE)):
            returnable = (kind == item) | ((station == "L") & (kind == CHOPPED_SIDE))
            held = torch.where(at(station) & empty, item,
                               torch.where(at(station) & returnable, EMPTY, held))

        pending = self.status == ACTIVE
        order = pending.long().argmax(1)
        cooker = at("G") & torch.isin(self.positions, self.cooker_ports)
        load = unique(cooker & (kind == RAW_MAIN) & ((self.cooker == EMPTY) & pending.any(1))[:, None])
        loading = load.any(1)
        cooking = (self.remaining > 0) & live
        ready = (self.cooker % ITEM_STRIDE == COOKED_MAIN) & (self.main == EMPTY) & live
        joint_ready = ready & self.joint[(self.cooker // ITEM_STRIDE - 1).clamp_min(0)]
        unloaders = cooker & empty & ready[:, None]
        count = unloaders.sum(1)
        unloading = torch.where(joint_ready, count == 2, count > 0)

        cutting = at("K")
        put_side = unique(cutting & (kind == RAW_SIDE) & (self.cutting == EMPTY)[:, None])
        take_side = unique(cutting & empty & (self.cutting == CHOPPED_SIDE)[:, None])
        chop = unique(cutting & empty & (self.cutting == RAW_SIDE)[:, None]).any(1)
        self.cutting = transfer(self.cutting, put_side, take_side)
        self.progress = torch.where(take_side.any(1), 0, self.progress + chop)
        self.cutting = torch.where(chop & (self.progress == self.config.chop_steps), CHOPPED_SIDE, self.cutting)

        buffering = at("B")
        self.buffer = transfer(self.buffer, unique(buffering & ~empty & (self.buffer == EMPTY)[:, None]),
                               unique(buffering & empty & (self.buffer != EMPTY)[:, None]))

        assembly = at("A") & torch.isin(self.positions, self.assembly_ports)
        assemble = unique(assembly & empty & ((self.main != EMPTY) & (self.side != EMPTY) &
                                              (self.plate == PLATE))[:, None]).any(1)
        take_meal = unique(assembly & empty & (self.plate % ITEM_STRIDE == MEAL)[:, None])
        put_plate = unique(assembly & ((kind == PLATE) | (kind == MEAL)) & (self.plate == EMPTY)[:, None])
        put_chopped = unique(assembly & (kind == CHOPPED_SIDE) & (self.side == EMPTY)[:, None])
        self.plate = transfer(self.plate, put_plate, take_meal)
        self.plate = torch.where(assemble, self.main - COOKED_MAIN + MEAL, self.plate)
        self.side = torch.where(assemble, EMPTY, deposit(self.side, put_chopped))
        self.main = torch.where(assemble, EMPTY, torch.where(unloading, self.cooker, self.main))
        self.cooker = torch.where(loading, (order + 1) * ITEM_STRIDE + RAW_MAIN,
                                  torch.where(unloading, EMPTY, self.cooker))
        self.remaining = torch.where(loading, self.config.cook_steps, self.remaining - cooking.long())
        self.cooker += (cooking & (self.remaining == 0)).long() * (COOKED_MAIN - RAW_MAIN)

        serve = at("S") & (kind == MEAL)
        delivered = (serve[:, :, None] & ((old // ITEM_STRIDE - 1)[:, :, None] == self.ids)).any(1)
        self.held = torch.where(put_chopped | load | serve, EMPTY, held)
        self._move(actions)
        self.tick += live
        starting = loading[:, None] & (order[:, None] == self.ids)
        self.started = torch.where(starting, self.tick[:, None], self.started)
        self.status = torch.where(starting, STARTED, torch.where(delivered, SERVED, self.status))
        queued = self.status == QUEUED
        capacity = self.config.active_limit - ((self.status == STARTED) | (self.status == ACTIVE)).sum(1)
        release = queued & (queued.cumsum(1) <= capacity[:, None]) & (self.tick < self.config.horizon)[:, None]
        self.status = torch.where(release, ACTIVE, self.status)
        self.released = torch.where(release, self.tick[:, None], self.released)

        normal_count = (delivered & ~self.joint).sum(1)
        joint_count = (delivered & self.joint).sum(1)
        reward = normal_count.double() * self.config.normal_reward + joint_count.double() * self.config.joint_reward
        terminated = (self.status == SERVED).all(1)
        truncated = (self.tick >= self.config.horizon) & ~terminated
        info = {"normal_deliveries": normal_count, "joint_deliveries": joint_count,
                "joint_unload": joint_ready & unloading, "unloaders": unloaders & unloading[:, None],
                "chopped": chop}
        return self.observations(), reward, terminated, truncated, info
