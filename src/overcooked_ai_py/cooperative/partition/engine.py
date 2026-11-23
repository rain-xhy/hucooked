"""Batched partition rules. NumPy runs the native reference, Torch runs device lanes."""
import numpy as np

from .model import (BITS, BOARDS, CUT_POTATO, CUT_TOMATO, DELIVERY_SIDE, EMPTY, FIRST_MEAL,
                    FLOOR, FLOORS, HEIGHT, INTERACT, OFFSETS, POTATO, POTS, PartitionConfig,
                    RECIPES, SERVERS, SOURCES, STAY, TOMATO, WALL, WIDTH)


class BatchKitchen:
    def __init__(self, config, num_envs, device=None):
        self.config = config or PartitionConfig()
        if not isinstance(self.config, PartitionConfig) or type(num_envs) is not int or num_envs < 1:
            raise ValueError("Expected a PartitionConfig and positive integer batch size")
        self.num_envs = num_envs
        self.device = device
        if device is None:
            self.xp = np
        else:
            import torch
            self.xp = torch
            self.device = torch.empty(0, device=device).device
        self.where = self.xp.where
        self.lanes = self.array(np.arange(num_envs, dtype=np.int64))
        self.slots = self.array(np.arange(3, dtype=np.int64))
        self.floor = self.array([i in FLOOR for i in range(WIDTH * HEIGHT)])
        self.offsets = self.array(OFFSETS)
        self.deadlines = self.array(self.config.deadlines)
        self.delivery_side = self.array(DELIVERY_SIDE)
        self.servers = self.array(SERVERS)
        self.bits = self.array([BITS.get(i, 0) for i in range(FIRST_MEAL + 3)])
        self.valid_mix = self.array([any((mix & recipe) == mix for recipe in RECIPES) for mix in range(16)])
        self.recipe_for_mix = self.array([RECIPES.index(i) if i in RECIPES else -1 for i in range(16)])
        self.state = {}
        for name, width in (("tick", 1), ("pos", 3), ("facing", 3), ("held", 3),
                            ("board", 4), ("cuts", 4), ("mix", 2), ("locked", 2), ("heat", 2),
                            ("counter", 5), ("deposited", 5), ("ids", 3), ("recipe", 3),
                            ("issued", 3), ("due", 3), ("next_id", 1)):
            self.state[name] = self.array(np.zeros((num_envs, width), dtype=np.int64))
        self.draws = self.array(np.zeros((num_envs, 3 + 3 * self.config.horizon), dtype=np.int64))
        self.rng = [None] * num_envs
        self.reset(seed=0)

    def array(self, value):
        if self.device is None:
            return np.asarray(value)
        return self.xp.as_tensor(value, device=self.device)

    def ints(self, values):
        return values.astype(np.int64) if self.device is None else values.long()

    def copy(self, values):
        return values.copy() if self.device is None else values.clone()

    def cat(self, values):
        if self.device is None:
            return np.concatenate(values, axis=1)
        return self.xp.cat(values, dim=1)

    def reset(self, seed=None, mask=None):
        indices = list(range(self.num_envs))
        if mask is not None:
            flags = self.array(mask)
            if flags.shape != (self.num_envs,) or str(flags.dtype) not in ("bool", "torch.bool"):
                raise ValueError("Expected a boolean reset flag per lane")
            indices = [i for i, flag in enumerate(flags.tolist()) if flag]
        if indices:
            pos, facing, draws = [], [], []
            for i in indices:
                if seed is not None or self.rng[i] is None:
                    streams = np.random.SeedSequence(None if seed is None else seed + i).spawn(2)
                    self.rng[i] = tuple(np.random.default_rng(s) for s in streams)
                spawn_rng, ticket_rng = self.rng[i]
                cells = [spawn_rng.choice(FLOORS[0]), *spawn_rng.choice(FLOORS[1], 2, replace=False)]
                pos.append(spawn_rng.permutation(cells))
                facing.append(spawn_rng.integers(4, size=3))
                draws.append(ticket_rng.integers(3, size=self.draws.shape[1]))
            idx = self.array(indices)
            for value in self.state.values():
                value[idx] = 0
            self.draws[idx] = self.array(np.stack(draws))
            s = self.state
            s["pos"][idx] = self.array(np.stack(pos).astype(np.int64))
            s["facing"][idx] = self.array(np.stack(facing))
            s["locked"][idx] = -1
            s["ids"][idx] = self.slots
            s["recipe"][idx] = self.draws[idx, :3]
            s["due"][idx] = self.deadlines[s["recipe"][idx]]
            s["next_id"][idx] = 3
        return self.observations(), {}

    def observations(self):
        s = self.state
        def scaled(key, scale):
            return s[key] / scale
        fields = [s["tick"] / self.config.horizon, s["tick"] * 0 + int(self.config.layout == "sync"),
                  scaled("pos", WIDTH * HEIGHT - 1), scaled("facing", 3), scaled("held", 9),
                  scaled("board", 6), scaled("cuts", 2)]
        fields.extend(self.ints((s["mix"] & bit) != 0) for bit in (1, 2, 4, 8))
        fields.extend(((s["locked"] + 1) / 3, scaled("heat", 4), scaled("counter", 9),
                       scaled("recipe", 2), (s["due"] - s["tick"]) / max(self.config.deadlines),
                       self.delivery_side[s["recipe"]]))
        common = self.cat(fields)
        common = common.astype(np.float32) if self.device is None else common.float()
        views = [self.cat((common, common[:, :1] * 0 + i / 2)) for i in range(3)]
        return self.xp.stack(views, axis=1)

    def _resolve_movement(self, actions):
        s = self.state
        moving = actions < STAY
        s["facing"] = self.where(moving, actions, s["facing"])
        dest = s["pos"] + self.offsets[s["facing"]]
        desired = self.where(moving & self.floor[dest], dest, s["pos"])
        pairs = self.slots[:, None] != self.slots
        follows = (desired[:, :, None] == s["pos"][:, None, :]) & pairs
        swap = follows & follows.transpose(0, 2, 1) if self.device is None else follows & follows.transpose(1, 2)
        blocked = (((desired[:, :, None] == desired[:, None, :]) & pairs) | swap).any(axis=2)
        for _ in range(2):
            blocked = blocked | (follows & blocked[:, None, :]).any(axis=2)
        s["pos"] = self.where(blocked, s["pos"], desired)

    def _boards(self, target, hand, live):
        s = self.state
        for j, station in enumerate(BOARDS):
            at = target == station
            item = self.copy(s["board"][:, j])
            progress = s["cuts"][:, j]
            put = at & ((hand == TOMATO) | (hand == POTATO)) & (item == EMPTY)[:, None]
            put = put & (put.sum(axis=1) == 1)[:, None]
            empty = at & (hand == EMPTY)
            take = empty & (progress == 2)[:, None] & (empty.sum(axis=1) == 1)[:, None]
            cutting = empty.any(axis=1) & (item != EMPTY) & (progress < 2)
            updated = self.where(cutting, progress + 1, self.where(progress == 2, 2, 0))
            s["cuts"][:, j] = self.where(live, self.where(take.any(axis=1), 0, updated), progress)
            s["board"][:, j] = self.where(put.any(axis=1), self.where(put, hand, 0).sum(axis=1),
                                          self.where(take.any(axis=1), EMPTY, item))
            chopped = self.where(item == TOMATO, CUT_TOMATO, CUT_POTATO)
            s["held"] = self.where(put, EMPTY, self.where(take, chopped[:, None], s["held"]))

    def _pots(self, target, hand, live):
        s = self.state
        bit = self.bits[hand]
        for j, station in enumerate(POTS):
            at = target == station
            mix = self.copy(s["mix"][:, j])
            lock = self.copy(s["locked"][:, j])
            heat = s["heat"][:, j]
            candidate = mix[:, None] | bit
            put = at & (bit != 0) & (lock < 0)[:, None] & ((mix[:, None] & bit) == 0) & self.valid_mix[candidate]
            empty = at & (hand == EMPTY)
            matched = self.recipe_for_mix[mix]
            tend = empty.any(axis=1) & (matched >= 0) & (heat < 4)
            # A deposit and ignition competing for the same pot both wait.
            competing = tend & put.any(axis=1)
            tend = tend & ~competing
            put = put & ((put.sum(axis=1) == 1) & ~competing)[:, None]
            take = empty & (heat == 4)[:, None] & (empty.sum(axis=1) == 1)[:, None]
            taken = take.any(axis=1)
            updated = self.where(tend, heat + 1, self.where(heat == 4, 4, 0))
            s["heat"][:, j] = self.where(live, self.where(taken, 0, updated), heat)
            s["locked"][:, j] = self.where(taken, -1, self.where(tend, matched, lock))
            s["mix"][:, j] = self.where(taken, 0, mix | self.where(put, bit, 0).sum(axis=1))
            s["held"] = self.where(put, EMPTY, self.where(take, (FIRST_MEAL + lock)[:, None], s["held"]))

    def _handoffs(self, target, hand, tick):
        s = self.state
        attempts, successes, waits, pickups = [tick * 0 for _ in range(4)]
        for j, station in enumerate(WALL):
            at = target == station
            giving = at & (hand != EMPTY)
            receiving = at & (hand == EMPTY)
            item = self.where(giving, hand, 0).sum(axis=1)
            if self.config.layout == "sync":
                if j not in (1, 3):
                    continue
                paired = giving.any(axis=1) & receiving.any(axis=1)
                attempts += self.ints(at.any(axis=1))
                successes += self.ints(paired)
                s["held"] = self.where(giving & paired[:, None], EMPTY,
                                       self.where(receiving & paired[:, None], item[:, None], s["held"]))
            else:
                stored = self.copy(s["counter"][:, j])
                single = at.sum(axis=1) == 1
                put = single & giving.any(axis=1) & (stored == EMPTY)
                take = single & receiving.any(axis=1) & (stored != EMPTY)
                waits += self.where(take, tick - s["deposited"][:, j], 0)
                pickups += self.ints(take)
                s["deposited"][:, j] = self.where(put, tick, s["deposited"][:, j])
                s["counter"][:, j] = self.where(put, item, self.where(take, EMPTY, stored))
                s["held"] = self.where(giving & put[:, None], EMPTY,
                                       self.where(receiving & take[:, None], stored[:, None], s["held"]))
        return dict(sync_attempts=attempts, sync_successes=successes, wait_steps=waits, pickups=pickups)

    def _orders(self, target, hand, tick, live):
        s = self.state
        recipe = s["recipe"]
        delivered = (s["ids"] < 0)
        for chef in range(3):
            match = (recipe == (hand[:, chef] - FIRST_MEAL)[:, None]) & ~delivered
            side = target[:, chef, None] == self.servers[self.delivery_side[recipe]]
            match = match & side
            first = match & (self.ints(match).cumsum(axis=1) == 1)
            delivered = delivered | first
            s["held"][:, chef] = self.where(first.any(axis=1), EMPTY, s["held"][:, chef])
        expired = (s["due"] <= tick[:, None]) & live[:, None] & ~delivered
        info = dict(recipe=recipe, order_ids=s["ids"], age=tick[:, None] - s["issued"],
                    delivered=delivered, expired=expired)
        removed = delivered | expired
        # Sort survivors by original slot, then append as many fresh tickets as removed.
        perm = (self.slots + 3 * self.ints(removed)).argsort(axis=1)
        retained = 3 - removed.sum(axis=1)
        fresh = self.slots >= retained[:, None]
        new_ids = s["next_id"] + self.slots - retained[:, None]
        safe_ids = self.where(fresh, new_ids, 0)
        new_recipe = self.draws[self.lanes[:, None], safe_ids]
        rows = self.lanes[:, None]
        for key, values in (("ids", new_ids), ("recipe", new_recipe), ("issued", tick[:, None]),
                            ("due", tick[:, None] + self.deadlines[new_recipe])):
            s[key] = self.where(fresh, values, s[key][rows, perm])
        s["next_id"] += removed.sum(axis=1)[:, None]
        return info

    def step(self, actions):
        a = self.array(actions)
        if a.shape != (self.num_envs, 3) or str(a.dtype).split(".")[-1] not in ("int8", "int16", "int32", "int64", "uint8"):
            raise ValueError("Expected three integer actions per lane")
        if ((a < 0) | (a > INTERACT)).any().item():
            raise ValueError("Invalid action")
        s = self.state
        live = s["tick"][:, 0] < self.config.horizon
        a = self.where(live[:, None], self.ints(a), STAY)
        tick = s["tick"][:, 0] + self.ints(live)
        target = self.where(a == INTERACT, s["pos"] + self.offsets[s["facing"]], -1)
        hand = self.copy(s["held"])
        for item, station in SOURCES.items():
            s["held"] = self.where((target == station) & (hand == EMPTY), item, s["held"])
        self._boards(target, hand, live)
        self._pots(target, hand, live)
        info = self._handoffs(target, hand, tick)
        info.update(self._orders(target, hand, tick, live))
        self._resolve_movement(a)
        s["tick"] = tick[:, None]
        count = info["delivered"].sum(axis=1)
        reward = (count.astype(np.float64) if self.device is None else count.double()) * self.config.delivery_reward
        return self.observations(), reward, live & False, tick >= self.config.horizon, info
