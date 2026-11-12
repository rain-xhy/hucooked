"""Batched tensor execution of the fixed-order MixedKitchen task."""

import copy

import numpy as np
import torch

from .kitchen import DIRECTIONS, INTERACT, STAY, VERSION, MixedKitchen
from .model import KitchenState
from .torch_state import (
    ACTIVE, AGENT_METRIC_INDEX, CERTIFIED, COMPLETED, COMPLETED_AT, COOK, CREATED,
    DISH, DUE, EMPTY, EXPIRED, INGREDIENT, ITEM_CERTIFIED, ITEM_KIND, ITEM_ORDER,
    JOINT, METRIC_INDEX, NORMAL, ORDER_ID, POT_ORDER, REMAINING, SOUP, STATUS,
    STOCK, SYNC, SYNC_DUE, TorchKitchenInfo, TorchKitchenState, decode_state,
    encode_order, encode_state,
)


class TorchMixedKitchen:
    """Independent kitchens with one layout and rule configuration per batch.

    Transitions and observations operate on tensors. Reset, snapshots, and
    rendering use the native model at an explicit CPU boundary.
    """

    def __init__(self, config=None, *, device="cuda", num_envs=1):
        if type(num_envs) is not int or num_envs < 1:
            raise ValueError("num_envs must be a positive integer")
        self._native = MixedKitchen(config)
        self.config = self._native.config
        self.active = self._native.active
        self.fingerprint = self._native.fingerprint
        self.device = torch.empty(0, device=device).device
        self.num_envs = num_envs
        self.width = len(self._native.terrain[0])
        self.height = len(self._native.terrain)
        self._batches = min(self.config.max_batches, self.config.horizon)
        self._history_capacity = 2 * self._batches
        self._rows = torch.arange(num_envs, device=self.device)
        self._players = torch.arange(3, device=self.device)
        self._active = self._tensor([i in self.active for i in range(3)], dtype=torch.bool)
        self._bits = self._tensor([1, 2, 4])
        self._view_order = self._tensor([[0, 1, 2], [1, 0, 2], [2, 0, 1]])
        self._order_kind = torch.eye(2, dtype=torch.float32, device=self.device).expand(num_envs, -1, -1)
        self._deadline_fields = self._tensor([DUE, SYNC_DUE])
        self._duration_fields = self._tensor([COOK, SYNC])
        self._empty = self._tensor([EMPTY, -1, 0])
        self._dish = self._tensor([DISH, -1, 0])
        self._ingredients = self._tensor([[INGREDIENT, NORMAL, 0], [INGREDIENT, JOINT, 0]])

        cells = torch.arange(self.width * self.height, device=self.device)
        xy = torch.stack((cells % self.width, cells // self.width), dim=1)
        neighbors = xy[:, None] + self._tensor(DIRECTIONS)
        neighbors = neighbors.clamp_min(0).minimum(self._tensor([self.width - 1, self.height - 1]))
        self._neighbors = neighbors[..., 0] + self.width * neighbors[..., 1]
        floor = [(x, y) in self._native.floor for y in range(self.height) for x in range(self.width)]
        self._floor = self._tensor(floor, dtype=torch.bool)
        locations = self._native.locations
        self._sources = [self._flat(locations[key][0]) for key in ("T", "O")]
        self._handoff = self._flat(locations["H"][0])
        self._dish_source = self._flat(locations["D"][0])
        self._serving = self._flat(locations["S"][0])
        self._pots = [self._flat(position) for position in self._native.pot_positions]
        self._buffers = [self._flat(position) for position in self._native.buffer_positions]
        facilities = [locations[key][0] for key in ("O", "T", "D", "S", "H")]
        facilities += self._native.pot_positions + self._native.buffer_positions
        self._static = self._tensor(
            [*floor, *(coordinate for x, y in facilities for coordinate in (x / 10, y / 6))],
            dtype=torch.float32,
        )
        self._time_scale = max(self.config.horizon, self.config.normal_deadline,
                               self.config.joint_deadline) + self.config.deadline_jitter
        self._duration_scale = max(64, self.config.cook_steps + self.config.duration_jitter,
                                   self.config.sync_steps + self.config.duration_jitter)
        self._reward = self._tensor([self.config.normal_reward, self.config.joint_reward], dtype=torch.float64)
        self._rng_states = [None] * num_envs
        self.state = None
        self.reset(seed=0)

    def _tensor(self, values, dtype=torch.long):
        return torch.tensor(values, dtype=dtype, device=self.device)

    def _flat(self, position):
        return position[0] + self.width * position[1]

    def _encode_lane(self, state, rng):
        encoded = encode_state(state, self.width, self._history_capacity)
        # Replay native RNG draws ahead of time so spawning a batch during
        # step needs neither host RNG nor a device-to-host synchronization.
        self._native.rng.bit_generator.state = copy.deepcopy(rng)
        schedule = [[encode_order(order) for order in state.orders]]
        rng_states = [copy.deepcopy(rng)]
        scratch = KitchenState([], [], [], batch=state.batch,
                               metrics={"normal_issued": 0, "joint_issued": 0})
        remaining = min(self.config.max_batches - state.batch, self.config.horizon - state.tick,
                        self._batches - 1)
        for _ in range(remaining):
            self._native._spawn_batch(scratch)
            schedule.append([encode_order(order) for order in scratch.orders])
            rng_states.append(copy.deepcopy(self._native.rng.bit_generator.state))
        return encoded, schedule + [schedule[-1]] * (self._batches - len(schedule)), rng_states

    def _replace_lanes(self, indices, lanes):
        states, schedules, rng_states = zip(*lanes)
        dtypes = {"score": torch.float64, "terminated": torch.bool,
                  "truncated": torch.bool, "pair": torch.bool}
        new = TorchKitchenState(*(
            self._tensor(field, dtype=dtypes.get(name, torch.long))
            for name, field in zip(TorchKitchenState._fields, zip(*states))
        ))
        schedule = self._tensor(schedules)
        if self.state is None:
            self.state, self._schedule = new, schedule
        else:
            rows = self._tensor(indices)
            self.state = TorchKitchenState(*(old.index_copy(0, rows, field) for old, field in zip(self.state, new)))
            self._schedule = self._schedule.index_copy(0, rows, schedule)
        for lane, rng in zip(indices, rng_states):
            self._rng_states[lane] = rng

    def reset(self, seed=None, *, mask=None):
        """Reset selected lanes. An integer seed uses seed + lane for each kitchen."""
        if mask is None:
            indices = list(range(self.num_envs))
        else:
            mask = torch.as_tensor(mask, dtype=torch.bool)
            if mask.shape != (self.num_envs,):
                raise ValueError(f"Expected reset mask of shape {(self.num_envs,)}")
            indices = [lane for lane, selected in enumerate(mask.tolist()) if selected]
        if indices:
            schedule_indices = self.state.schedule_index.tolist() if self.state is not None else None
            lanes = []
            for lane in indices:
                if seed is None and schedule_indices is not None:
                    self._native.rng.bit_generator.state = self._rng_states[lane][schedule_indices[lane]]
                self._native.reset(None if seed is None else seed + lane)
                lanes.append(self._encode_lane(self._native.state, self._native.rng.bit_generator.state))
            self._replace_lanes(indices, lanes)
        zero = torch.zeros_like(self.state.score)
        return self.observations(), self._info((), zero, zero)

    def to_native_state(self, lane=0):
        return decode_state(self.state, lane, self.width)

    def snapshot(self, lane=0):
        state = self.to_native_state(lane)
        schedule_index = self.state.schedule_index[lane].item()
        return {"version": VERSION, "config": self.config.to_dict(),
                "fingerprint": self.fingerprint, "state": state.to_dict(),
                "rng": copy.deepcopy(self._rng_states[lane][schedule_index])}

    def restore(self, snapshot, *, lane=0):
        """Load a compatible native or tensor snapshot into one lane."""
        if not 0 <= lane < self.num_envs:
            raise IndexError("Lane is outside the batch")
        self._native.restore(snapshot)
        encoded = self._encode_lane(self._native.state, self._native.rng.bit_generator.state)
        self._replace_lanes([lane], [encoded])

    def render(self, lane=0):
        renderer = copy.copy(self._native)
        renderer.state = self.to_native_state(lane)
        return renderer.render()

    def _info(self, events, raw, shaping):
        state = self.state
        return TorchKitchenInfo(state.tick, state.batch, state.score, raw, shaping, state.metrics,
                                state.agent_metrics, state.latencies, tuple(events),
                                self.fingerprint, self.width, self.active)

    def _validated_actions(self, actions):
        if not isinstance(actions, torch.Tensor):
            values = np.asarray(actions, dtype=object)
            if values.shape != (self.num_envs, 3) or any(
                isinstance(action, (bool, np.bool_)) or not isinstance(action, (int, np.integer))
                or not 0 <= action <= INTERACT
                for action in values.flat
            ):
                raise ValueError("Expected three integer actions in [0, 5] per kitchen")
            return self._tensor(values.tolist())
        if (actions.shape != (self.num_envs, 3) or actions.dtype not in
                (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8)):
            raise ValueError("Expected three integer actions in [0, 5] per kitchen")
        if ((actions < 0) | (actions > INTERACT)).any():
            raise ValueError("Expected three integer actions in [0, 5] per kitchen")
        return actions.to(device=self.device, dtype=torch.long)

    @staticmethod
    def _metric(state, name):
        return state.metrics[:, METRIC_INDEX[name]]

    @staticmethod
    def _agent_metric(state, name):
        return state.agent_metrics[:, AGENT_METRIC_INDEX[name]]

    @torch.no_grad()
    def step(self, actions):
        """Advance unfinished lanes using an integer tensor of shape [batch, 3]."""
        actions = self._validated_actions(actions)
        before = self.state
        state = TorchKitchenState(*(field.clone() for field in before))
        live = ~(before.terminated | before.truncated)
        actions = torch.where(live[:, None] & self._active, actions, STAY)
        targets = self._neighbors[state.positions, state.orientations]
        events = []

        def record(kind, mask, *values):
            fields = tuple(value.clone() if isinstance(value, torch.Tensor) else
                           torch.full_like(state.tick, value) for value in values)
            events.append((kind, mask, fields))

        self._joint_interaction(state, actions, targets, live, record)
        self._ordinary_interactions(state, actions, targets, record)
        self._move(state, actions, record)
        state.tick.add_(live)
        state.pots[..., REMAINING].sub_(live[:, None].long()).clamp_min_(0)
        self._expire_orders(state, live, record)
        self._advance_batches(state, live, record)
        raw = state.score - before.score
        shaping = torch.zeros_like(raw)
        if self.config.shaping_scale:
            potential = self._potential(state).masked_fill(state.terminated, 0)
            shaping = self.config.shaping_scale * (self.config.shaping_gamma * potential - self._potential(before))
            shaping = shaping.masked_fill(~live, 0)
        self.state = state
        return self.observations(), raw + shaping, state.terminated, state.truncated, self._info(events, raw, shaping)

    def _joint_interaction(self, state, actions, targets, live, record):
        facing = (targets == self._handoff) & self._active & live[:, None]
        pressing = facing & (actions == INTERACT)
        holders = pressing & (state.held[..., ITEM_KIND] != EMPTY)
        donor = holders.long().argmax(dim=1)
        receiver = (pressing & ~holders).long().argmax(dim=1)
        item = state.held[self._rows, donor]
        order = state.orders[:, JOINT]
        valid = (
            (pressing.sum(dim=1) == 2) & (holders.sum(dim=1) == 1)
            & (item[:, ITEM_KIND] == SOUP) & (item[:, ITEM_ORDER] == JOINT)
            & (item[:, ITEM_CERTIFIED] == 0) & (order[:, STATUS] == ACTIVE)
            & (state.tick + 1 <= order[:, SYNC_DUE])
        )
        continuing = valid & (state.progress > 0) & (state.pair == pressing).all(dim=1)
        aborted = live & (state.progress > 0) & ~continuing
        started = valid & ~continuing
        progress = torch.where(continuing, state.progress, 0) + 1
        completed = valid & (progress == order[:, SYNC])
        waiting = live & ~valid
        invalid_press = waiting & pressing.any(dim=1)
        pair_bits = (pressing * self._bits).sum(dim=1)

        record("joint_aborted", aborted, order[:, ORDER_ID], state.progress)
        record("joint_started", started, order[:, ORDER_ID], pair_bits)
        record("joint_progress", valid, progress, order[:, SYNC], pair_bits)
        record("joint_completed", completed, order[:, ORDER_ID], donor, receiver)
        record("joint_not_ready", invalid_press, pair_bits)
        self._metric(state, "joint_aborts").add_(aborted)
        self._metric(state, "joint_starts").add_(started)
        self._metric(state, "joint_completions").add_(completed)
        self._metric(state, "invalid_joint_presses").add_(invalid_press)
        self._metric(state, "wait_agent_steps").add_(
            (facing & waiting[:, None] & (actions >= STAY)).sum(dim=1)
        )
        self._agent_metric(state, "joint_ticks_by_agent").add_(pressing & valid[:, None])
        order[:, CERTIFIED].masked_fill_(completed, 1)
        item[:, ITEM_CERTIFIED] = 1
        give = completed[:, None] & (self._players == donor[:, None])
        receive = completed[:, None] & (self._players == receiver[:, None])
        remaining = torch.where(give[..., None], self._empty, state.held)
        state.held.copy_(torch.where(receive[..., None], item[:, None], remaining))

        pending = valid & ~completed
        state.progress.copy_(torch.where(live, torch.where(pending, progress, 0), state.progress))
        state.pair.copy_(torch.where(live[:, None], pressing & pending[:, None], state.pair))
        state.donor.copy_(torch.where(live, torch.where(pending, donor, -1), state.donor))

    def _ordinary_interactions(self, state, actions, targets, record):
        pressing = (actions == INTERACT) & (targets != self._handoff)
        next_tick = state.tick + 1
        earlier = []
        for player in self.active:
            target = targets[:, player]
            participants = pressing & (targets == target[:, None])
            conflict = participants.sum(dim=1) > 1
            first = pressing[:, player] & ~participants[:, earlier].any(dim=1)
            earlier.append(player)
            record("resource_conflict", first & conflict, (participants * self._bits).sum(dim=1), target)
            use = pressing[:, player] & ~conflict
            item = state.held[:, player].clone()
            kind, slot, certified = item.unbind(dim=1)
            empty = kind == EMPTY
            next_item = item

            for order_slot, source in enumerate(self._sources):
                order = state.orders[:, order_slot]
                at_source = use & (target == source)
                taken = (at_source & empty & (order[:, STATUS] == ACTIVE)
                         & (order[:, STOCK] > 0) & (next_tick <= order[:, DUE]))
                returned = at_source & (kind == INGREDIENT) & (slot == order_slot)
                order[:, STOCK].add_(returned.long() - taken.long())
                next_item = torch.where(taken[:, None], self._ingredients[order_slot], next_item)
                next_item = torch.where(returned[:, None], self._empty, next_item)
                record("ingredient_taken", taken, player, order[:, ORDER_ID])
                record("ingredient_returned", returned, player, order[:, ORDER_ID])

            at_dish = use & (target == self._dish_source)
            dish_returned = at_dish & (kind == DISH)
            next_item = torch.where((at_dish & empty)[:, None], self._dish, next_item)
            next_item = torch.where(dish_returned[:, None], self._empty, next_item)
            record("dish_returned", dish_returned, player)

            for index, location in enumerate(self._buffers):
                buffer = state.buffers[:, index]
                at_buffer = use & (target == location)
                buffer_empty = buffer[:, ITEM_KIND] == EMPTY
                take = at_buffer & empty & ~buffer_empty
                leave = at_buffer & ~empty & buffer_empty
                next_item = torch.where(take[:, None], buffer, next_item)
                next_item = torch.where(leave[:, None], self._empty, next_item)
                buffer.copy_(torch.where(leave[:, None], item,
                                         torch.where(take[:, None], self._empty, buffer)))

            order = state.orders[self._rows, slot.clamp_min(0)]
            for index, location in enumerate(self._pots):
                pot = state.pots[:, index]
                pot_slot = pot[:, POT_ORDER]
                pot_order = state.orders[self._rows, pot_slot.clamp_min(0)]
                at_pot = use & (target == location)
                loaded = at_pot & (kind == INGREDIENT) & (pot_slot < 0)
                plated = at_pot & (kind == DISH) & (pot_slot >= 0) & (pot[:, REMAINING] == 0)
                soup = torch.stack((torch.full_like(pot_slot, SOUP), pot_slot,
                                    torch.zeros_like(pot_slot)), dim=1)
                next_item = torch.where(plated[:, None], soup, next_item)
                next_item = torch.where(loaded[:, None], self._empty, next_item)
                pot[:, POT_ORDER].copy_(torch.where(loaded, slot, torch.where(plated, -1, pot_slot)))
                pot[:, REMAINING].copy_(torch.where(loaded, order[:, COOK] + 1, pot[:, REMAINING]))
                self._agent_metric(state, "loads_by_agent")[:, player].add_(loaded)
                record("pot_loaded", loaded, player, order[:, ORDER_ID])
                record("soup_plated", plated, player, pot_order[:, ORDER_ID])

            serving = use & (target == self._serving) & (kind == SOUP)
            rejected = serving & ((next_tick > order[:, DUE]) | ((slot == JOINT) & (certified == 0)))
            delivered = serving & ~rejected
            next_item = torch.where(delivered[:, None], self._empty, next_item)
            state.held[:, player].copy_(next_item)
            record("delivery_rejected", rejected, player, order[:, ORDER_ID])
            record("delivered", delivered, player, slot, order[:, ORDER_ID])
            self._deliver(state, player, slot.clamp_min(0), order, next_tick, delivered)

    def _deliver(self, state, player, slot, order, next_tick, delivered):
        normal_completed = self._metric(state, "normal_completed")
        joint_completed = self._metric(state, "joint_completed")
        latency_index = (normal_completed + joint_completed).clamp_max(self._history_capacity - 1)
        latency = next_tick - order[:, CREATED]
        state.latencies[self._rows, latency_index] = torch.where(
            delivered, latency, state.latencies[self._rows, latency_index]
        )
        order[:, STATUS].masked_fill_(delivered, COMPLETED)
        order[:, COMPLETED_AT].copy_(torch.where(delivered, next_tick, order[:, COMPLETED_AT]))
        state.orders[self._rows, slot] = order
        normal_completed.add_(delivered & (slot == NORMAL))
        joint_completed.add_(delivered & (slot == JOINT))
        self._agent_metric(state, "serves_by_agent")[:, player].add_(delivered)
        state.score.add_(torch.where(delivered, self._reward[slot], 0))

    def _move(self, state, actions, record):
        old = state.positions
        moving = actions < STAY
        state.orientations.copy_(torch.where(moving, actions, state.orientations))
        target = self._neighbors[old, state.orientations]
        intended = torch.where(moving & self._floor[target], target, old)
        same_destination = intended[:, :, None] == intended[:, None, :]
        entering = intended[:, :, None] == old[:, None, :]
        swaps = entering & entering.transpose(1, 2)
        pairs = (self._active[:, None] & self._active) & (self._players[:, None] != self._players)
        blocked = ((same_destination | swaps) & pairs).any(dim=2)
        # At most two additional chefs can join a blocked chain in a three-slot kitchen.
        for _ in range(2):
            blocked |= (entering & pairs & blocked[:, None, :]).any(dim=2)
        collisions = blocked & (intended != old)
        self._metric(state, "collisions").add_(collisions.sum(dim=1))
        record("collision", collisions.any(dim=1), (collisions * self._bits).sum(dim=1))
        old.copy_(torch.where(blocked, old, intended))

    def _expire_orders(self, state, live, record):
        for slot, metric in ((NORMAL, "normal_missed"), (JOINT, "joint_missed")):
            order = state.orders[:, slot]
            late = state.tick > order[:, DUE]
            if slot == JOINT:
                late |= (order[:, CERTIFIED] == 0) & (state.tick > order[:, SYNC_DUE])
            expired = live & (order[:, STATUS] == ACTIVE) & late
            record("order_expired", expired, order[:, ORDER_ID], slot)
            self._metric(state, metric).add_(expired)
            self._metric(state, "wasted_ingredients").add_(expired & (order[:, STOCK] == 0))
            state.score.add_(expired.double() * self.config.expiry_penalty)
            order[:, STOCK].masked_fill_(expired, 0)
            order[:, STATUS].masked_fill_(expired, EXPIRED)
            for items in (state.held, state.buffers):
                remove = expired[:, None] & (items[..., ITEM_ORDER] == slot)
                items.copy_(torch.where(remove[..., None], self._empty, items))
            remove_pot = expired[:, None] & (state.pots[..., POT_ORDER] == slot)
            state.pots[..., POT_ORDER].masked_fill_(remove_pot, -1)
            state.pots[..., REMAINING].masked_fill_(remove_pot, 0)
            if slot == JOINT:
                state.progress.masked_fill_(expired, 0)
                state.pair.masked_fill_(expired[:, None], False)
                state.donor.masked_fill_(expired, -1)

    def _advance_batches(self, state, live, record):
        statuses = state.orders[..., STATUS]
        resolved = live & (statuses != ACTIVE).all(dim=1)
        completed = resolved & (statuses == COMPLETED).all(dim=1)
        state.score.add_(completed.double() * self.config.batch_bonus)
        self._metric(state, "batches_completed").add_(completed)
        record("batch_completed", completed, state.batch)
        for slot in (NORMAL, JOINT):
            index = (state.history_count + slot).clamp_max(self._history_capacity - 1)
            state.history[self._rows, index] = torch.where(
                resolved[:, None], state.orders[:, slot], state.history[self._rows, index]
            )
        state.history_count.add_(2 * resolved)
        closing = live & (state.tick >= self.config.horizon)
        spawn = resolved & ~closing & (state.batch < self.config.max_batches)
        state.schedule_index.add_(spawn)
        new_orders = self._schedule[self._rows, state.schedule_index]
        new_orders[..., CREATED] = state.tick[:, None]
        new_orders[..., DUE].add_(state.tick[:, None])
        new_orders[:, JOINT, SYNC_DUE].add_(state.tick)
        state.orders.copy_(torch.where(spawn[:, None, None], new_orders, state.orders))
        state.batch.add_(spawn)
        for metric in ("normal_issued", "joint_issued"):
            self._metric(state, metric).add_(spawn)
        state.terminated.logical_or_(closing | (resolved & ~spawn))
        self._metric(state, "unfinished_at_close").copy_(torch.where(
            closing, (statuses == ACTIVE).sum(dim=1), self._metric(state, "unfinished_at_close")
        ))

    def _potential(self, state):
        items = torch.cat((state.held, state.buffers), dim=1)
        kind, slot, certified = items.unbind(dim=2)
        item_stage = torch.where(kind == INGREDIENT, 1, torch.where(certified.bool(), 5, 4))
        pot_stage = torch.where(state.pots[..., REMAINING] > 0, 2, 3)
        stages = []
        for order_slot in (NORMAL, JOINT):
            in_hand = torch.where(slot == order_slot, item_stage, 0).amax(dim=1)
            in_pot = torch.where(state.pots[..., POT_ORDER] == order_slot, pot_stage, 0).amax(dim=1)
            stages.append(torch.maximum(in_hand, in_pot).double())
        stages[JOINT] += state.progress.double() / state.orders[:, JOINT, SYNC]
        return (torch.stack(stages, dim=1) * (state.orders[..., STATUS] == ACTIVE)).sum(dim=1)

    @staticmethod
    def _encode_items(items):
        kind, slot, certified = items.unbind(dim=2)
        category = torch.where(kind == INGREDIENT, 2 + slot, torch.where(kind == SOUP, 4 + slot, kind))
        return torch.cat((torch.nn.functional.one_hot(category, num_classes=6), certified[..., None]), dim=2)

    def observations(self):
        """Return native-order float32 observations [batch, chef, 187]."""
        state = self.state
        positions = state.positions
        coords = torch.stack(((positions % self.width).double() / 10,
                              (positions // self.width).double() / 6), dim=2)
        player_features = torch.cat((
            self._active.expand(self.num_envs, -1)[..., None],
            coords.float(),
            torch.nn.functional.one_hot(state.orientations, num_classes=4),
            self._encode_items(state.held),
        ), dim=2)
        pot_features = torch.cat((
            torch.nn.functional.one_hot(state.pots[..., POT_ORDER] + 1, num_classes=3),
            (state.pots[..., REMAINING, None].double() / self._duration_scale).float(),
        ), dim=2)
        buffers = self._encode_items(state.buffers).flatten(1)
        orders = state.orders
        order_time = (orders.index_select(2, self._deadline_fields) - state.tick[:, None, None]).clamp_min(0)
        order_features = torch.cat((
            self._order_kind, torch.nn.functional.one_hot(orders[..., STATUS], num_classes=3),
            orders[..., STOCK, None],
            (order_time.double() / self._time_scale).float(),
            (orders.index_select(2, self._duration_fields).double() / self._duration_scale).float(),
            orders[..., CERTIFIED, None],
        ), dim=2)
        time_features = torch.stack((
            (self.config.horizon - state.tick).clamp_min(0).double() / self.config.horizon,
            state.batch.double() / self.config.max_batches,
            state.progress.double() / orders[:, JOINT, SYNC],
        ), dim=1).float()
        common = torch.cat((pot_features.flatten(1), buffers, order_features.flatten(1), time_features), dim=1)
        return torch.cat((
            self._static.expand(self.num_envs, 3, -1),
            player_features[:, self._view_order].flatten(2),
            common[:, None].expand(-1, 3, -1),
            state.pair[:, self._view_order],
        ), dim=2)

    def centralized_state(self):
        return self.observations()[:, 0]
