"""MixedKitchen-v1: finite mixed orders and sustained two-chef handoffs.

All dynamics use the current state and one simultaneous joint action. Rendering
is optional and reuses Overcooked's sprites; training never renders pixels.
"""

import copy
import hashlib
import json
from dataclasses import asdict
from functools import cached_property
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from .layouts import load_layout
from .model import Chef, Item, JointOperation, KitchenConfig, KitchenState, Order, Pot

DIRECTIONS = ((0, -1), (0, 1), (1, 0), (-1, 0))
STAY, INTERACT = 4, 5
VERSION = "MixedKitchen-v1"


def adjacent(position, direction):
    return (position[0] + direction[0], position[1] + direction[1])


class MixedKitchen:
    """Three homogeneous chef slots, six integer actions and shared reward.

    Action order matches Overcooked: north, south, east, west, stay, interact.
    Inactive slots are ghosted, permitting fair two-versus-three capability tests.
    """

    def __init__(self, config=None):
        self.config = config or KitchenConfig()
        self.terrain, self.locations, self.floor = load_layout(self.config.layout)
        self.handoff = self.locations["H"][0]
        self.sync_positions = (
            (self.handoff[0] - 1, self.handoff[1]),
            (self.handoff[0] + 1, self.handoff[1]),
        )
        self.pot_positions = self.locations["P"]
        self.buffer_positions = self.locations["B"]
        self.active = tuple(self.config.active_players)
        self.rng = np.random.default_rng()
        self.state = None
        self.reset(seed=0)

    @cached_property
    def fingerprint(self):
        source_hash = hashlib.sha256()
        for name in ("kitchen.py", "model.py", "layouts.py"):
            source_hash.update(Path(__file__).with_name(name).read_bytes().replace(b"\r\n", b"\n"))
        payload = {"version": VERSION, "config": self.config.to_dict(),
                   "terrain": self.terrain, "rules_sha256": source_hash.hexdigest()}
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]

    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        starts = [self.locations[str(i)][0] for i in (1, 2, 3)]
        if self.config.shuffle_players:
            starts = [starts[i] for i in self.rng.permutation(3)]
        self.state = KitchenState(
            chefs=[Chef(p) for p in starts], pots=[Pot(), Pot()], buffers=[None, None],
            metrics={"normal_completed": 0, "joint_completed": 0,
                     "normal_issued": 0, "joint_issued": 0, "batches_completed": 0,
                     "unfinished_at_close": 0,
                     "normal_missed": 0, "joint_missed": 0,
                     "joint_starts": 0, "joint_completions": 0, "joint_aborts": 0,
                     "invalid_joint_presses": 0, "wait_agent_steps": 0,
                     "wasted_ingredients": 0, "collisions": 0,
                     "loads_by_agent": [0, 0, 0], "serves_by_agent": [0, 0, 0],
                     "joint_ticks_by_agent": [0, 0, 0], "latencies": []},
        )
        self._spawn_batch(self.state)
        if self.config.reset_mode != "standard":
            self._prepared_reset()
        self.validate_state(self.state)
        return self.observations(), self._info([], 0.0, 0.0)

    def _prepared_reset(self):
        """Published curriculum distributions; never used for standard evaluation."""
        state = self.state
        active = list(self.active)
        if self.config.shuffle_players:
            active = [active[i] for i in self.rng.permutation(len(active))]
        for order in state.orders:
            order.stock = 0
        state.pots[1] = Pot(state.orders[0].id, 0)
        state.buffers[1] = Item("dish")
        if self.config.reset_mode == "kitchen_ready":
            state.pots[0] = Pot(state.orders[1].id, 0)
            state.buffers[0] = Item("dish")
        else:
            donor, receiver = active[:2]
            state.chefs[donor] = Chef((4, 3), (1, 0), Item("soup", state.orders[1].id))
            state.chefs[receiver] = Chef((6, 3), (-1, 0))
            if len(active) == 3:
                state.chefs[active[2]] = Chef((6, 4), (-1, 0))

    def _spawn_batch(self, state):
        state.batch += 1
        state.metrics["normal_issued"] += 1
        state.metrics["joint_issued"] += 1
        delta = int(self.rng.integers(-self.config.deadline_jitter,
                                      self.config.deadline_jitter + 1))
        def duration(value):
            return value + int(self.rng.integers(-self.config.duration_jitter,
                                                self.config.duration_jitter + 1))
        state.orders = [
            Order(2 * state.batch, "normal", state.tick,
                  state.tick + self.config.normal_deadline + delta, 0,
                  duration(self.config.cook_steps), 0),
            Order(2 * state.batch + 1, "joint", state.tick,
                  state.tick + self.config.joint_deadline + delta,
                  state.tick + self.config.sync_deadline + delta,
                  duration(self.config.cook_steps), duration(self.config.sync_steps)),
        ]
        state.joint = JointOperation()

    def order(self, order_id, state=None):
        state = state or self.state
        return next((o for o in state.orders if o.id == order_id), None)

    def item_type(self, item, state=None):
        if item is None:
            return 0
        if item.kind == "dish":
            return 1
        order = self.order(item.order_id, state)
        if order is None:
            raise ValueError("Item refers to an unknown order")
        return (2 if item.kind == "ingredient" else 4) + (order.kind == "joint")

    def snapshot(self):
        return {"version": VERSION, "config": self.config.to_dict(),
                "fingerprint": self.fingerprint,
                "state": self.state.to_dict(),
                "rng": copy.deepcopy(self.rng.bit_generator.state)}

    def restore(self, snapshot):
        if (snapshot["version"] != VERSION or KitchenConfig(**snapshot["config"]) != self.config
                or snapshot["fingerprint"] != self.fingerprint):
            raise ValueError("Snapshot version or task configuration differs")
        state = KitchenState.from_dict(copy.deepcopy(snapshot["state"]))
        self.validate_state(state)
        rng = np.random.default_rng()
        rng.bit_generator.state = copy.deepcopy(snapshot["rng"])
        self.state, self.rng = state, rng

    def step(self, actions):
        if self.state.terminated or self.state.truncated:
            raise RuntimeError("Reset the environment after episode end")
        if len(actions) != 3 or any(
            isinstance(a, (bool, np.bool_)) or not isinstance(a, (int, np.integer))
            or not 0 <= int(a) < 6 for a in actions
        ):
            raise ValueError("Expected three integer actions in [0, 5]")
        actions = tuple(int(a) if i in self.active else STAY for i, a in enumerate(actions))
        old_potential = self.potential(self.state)
        state = copy.deepcopy(self.state)
        old_score = state.score
        events = []
        self._joint_interaction(state, actions, events)
        self._ordinary_interactions(state, actions, events)
        self._move(state, actions, events)
        state.tick += 1
        for pot in state.pots:
            if pot.order_id is not None:
                pot.remaining = max(0, pot.remaining - 1)
        self._expire_orders(state, events)
        if all(o.status != "active" for o in state.orders):
            if all(o.status == "completed" for o in state.orders):
                state.score += self.config.batch_bonus
                state.metrics["batches_completed"] += 1
                events.append({"type": "batch_completed", "batch": state.batch})
            state.history.extend(asdict(o) for o in state.orders)
            if state.batch < self.config.max_batches and state.tick < self.config.horizon:
                self._spawn_batch(state)
            else:
                state.terminated = True
        # Closing time is part of this finite-horizon task, not an external cutoff.
        if state.tick >= self.config.horizon:
            state.terminated = True
            state.metrics["unfinished_at_close"] = sum(o.status == "active" for o in state.orders)
        self.validate_state(state)
        raw = state.score - old_score
        next_potential = 0.0 if state.terminated else self.potential(state)
        shaping = self.config.shaping_scale * (
            self.config.shaping_gamma * next_potential - old_potential
        )
        self.state = state
        return (self.observations(), raw + shaping, state.terminated,
                state.truncated, self._info(events, raw, shaping))

    def _joint_interaction(self, state, actions, events):
        facing = [i for i in self.active
                  if adjacent(state.chefs[i].position, state.chefs[i].orientation) == self.handoff]
        pressing = [i for i in facing if actions[i] == INTERACT]
        holders = [i for i in pressing if state.chefs[i].held is not None]
        valid = len(pressing) == 2 and len(holders) == 1
        donor = holders[0] if len(holders) == 1 else None
        item = state.chefs[donor].held if donor is not None else None
        order = self.order(item.order_id, state) if item is not None else None
        valid = valid and item.kind == "soup" and not item.certified and (
            order is not None and order.kind == "joint" and order.status == "active"
            and state.tick + 1 <= order.sync_due_at
        ) if valid else False
        pair = tuple(sorted(pressing))
        current = state.joint
        continuing = valid and current.order_id == order.id and current.pair == pair
        if current.progress and not continuing:
            state.metrics["joint_aborts"] += 1
            events.append({"type": "joint_aborted", "order_id": current.order_id,
                           "progress_lost": current.progress})
            state.joint = JointOperation()
        if valid:
            if not continuing:
                state.joint = JointOperation(order.id, pair, donor, 0)
                state.metrics["joint_starts"] += 1
                events.append({"type": "joint_started", "order_id": order.id, "pair": pair})
            state.joint.progress += 1
            for i in pair:
                state.metrics["joint_ticks_by_agent"][i] += 1
            events.append({"type": "joint_progress", "progress": state.joint.progress,
                           "required": order.sync_steps, "pair": pair})
            if state.joint.progress == order.sync_steps:
                receiver = next(i for i in pair if i != donor)
                item.certified = True
                order.certified = True
                state.chefs[receiver].held = item
                state.chefs[donor].held = None
                state.metrics["joint_completions"] += 1
                events.append({"type": "joint_completed", "order_id": order.id,
                               "donor": donor, "receiver": receiver})
                state.joint = JointOperation()
        else:
            if pressing:
                state.metrics["invalid_joint_presses"] += 1
                events.append({"type": "joint_not_ready", "agents": pressing})
            state.metrics["wait_agent_steps"] += sum(actions[i] in (STAY, INTERACT) for i in facing)

    def _ordinary_interactions(self, state, actions, events):
        targets = {}
        for i in self.active:
            if actions[i] == INTERACT:
                chef = state.chefs[i]
                target = adjacent(chef.position, chef.orientation)
                if target != self.handoff:
                    targets.setdefault(target, []).append(i)
        for target, agents in targets.items():
            # Contested ordinary facilities do nothing, avoiding player-index bias.
            if len(agents) != 1:
                events.append({"type": "resource_conflict", "agents": agents, "position": target})
                continue
            i = agents[0]
            chef = state.chefs[i]
            if target in (self.locations["O"][0], self.locations["T"][0]):
                kind = "joint" if target == self.locations["O"][0] else "normal"
                order = next(o for o in state.orders if o.kind == kind)
                if chef.held is None and order.status == "active" and order.stock and state.tick + 1 <= order.due_at:
                    chef.held = Item("ingredient", order.id)
                    order.stock -= 1
                    events.append({"type": "ingredient_taken", "agent": i, "order_id": order.id})
                elif chef.held is not None and chef.held.kind == "ingredient" and chef.held.order_id == order.id:
                    order.stock += 1
                    chef.held = None
                    events.append({"type": "ingredient_returned", "agent": i, "order_id": order.id})
            elif target == self.locations["D"][0]:
                if chef.held is None:
                    chef.held = Item("dish")
                elif chef.held.kind == "dish":
                    chef.held = None
                    events.append({"type": "dish_returned", "agent": i})
            elif target in self.buffer_positions:
                index = self.buffer_positions.index(target)
                if chef.held is not None and state.buffers[index] is None:
                    state.buffers[index], chef.held = chef.held, None
                elif chef.held is None and state.buffers[index] is not None:
                    chef.held, state.buffers[index] = state.buffers[index], None
            elif target in self.pot_positions:
                pot = state.pots[self.pot_positions.index(target)]
                if chef.held is not None and chef.held.kind == "ingredient" and pot.order_id is None:
                    order = self.order(chef.held.order_id, state)
                    pot.order_id, pot.remaining = order.id, order.cook_steps + 1
                    chef.held = None
                    state.metrics["loads_by_agent"][i] += 1
                    events.append({"type": "pot_loaded", "agent": i, "order_id": order.id})
                elif chef.held is not None and chef.held.kind == "dish" and pot.order_id is not None and pot.remaining == 0:
                    chef.held = Item("soup", pot.order_id)
                    pot.order_id, pot.remaining = None, 0
                    events.append({"type": "soup_plated", "agent": i, "order_id": chef.held.order_id})
            elif target == self.locations["S"][0] and chef.held is not None and chef.held.kind == "soup":
                order = self.order(chef.held.order_id, state)
                if state.tick + 1 > order.due_at or (order.kind == "joint" and not chef.held.certified):
                    events.append({"type": "delivery_rejected", "agent": i, "order_id": order.id})
                    continue
                order.status, order.completed_at = "completed", state.tick + 1
                chef.held = None
                state.score += self.config.normal_reward if order.kind == "normal" else self.config.joint_reward
                state.metrics[order.kind + "_completed"] += 1
                state.metrics["serves_by_agent"][i] += 1
                state.metrics["latencies"].append(order.completed_at - order.created_at)
                events.append({"type": "delivered", "agent": i, "kind": order.kind, "order_id": order.id})

    def _move(self, state, actions, events):
        old = {i: state.chefs[i].position for i in self.active}
        intended = dict(old)
        for i in self.active:
            if actions[i] < 4:
                direction = DIRECTIONS[actions[i]]
                state.chefs[i].orientation = direction
                target = adjacent(old[i], direction)
                if target in self.floor:
                    intended[i] = target
        blocked = set()
        for i in self.active:
            for j in self.active:
                if i >= j:
                    continue
                if intended[i] == intended[j] or (
                    intended[i] == old[j] and intended[j] == old[i]
                ):
                    blocked.update((i, j))
        changed = True
        while changed:
            changed = False
            for i in self.active:
                if i not in blocked and any(intended[i] == old[j] for j in blocked):
                    blocked.add(i)
                    changed = True
        collisions = [i for i in blocked if intended[i] != old[i]]
        state.metrics["collisions"] += len(collisions)
        if collisions:
            events.append({"type": "collision", "agents": sorted(collisions)})
        for i in self.active:
            state.chefs[i].position = old[i] if i in blocked else intended[i]

    def _expire_orders(self, state, events):
        for order in state.orders:
            if order.status != "active":
                continue
            if state.tick <= order.due_at and (
                order.kind != "joint" or order.certified or state.tick <= order.sync_due_at
            ):
                continue
            order.status = "expired"
            state.score += self.config.expiry_penalty
            state.metrics[order.kind + "_missed"] += 1
            if order.stock == 0:
                state.metrics["wasted_ingredients"] += 1
            order.stock = 0
            for chef in state.chefs:
                if chef.held is not None and chef.held.order_id == order.id:
                    chef.held = None
            for j, item in enumerate(state.buffers):
                if item is not None and item.order_id == order.id:
                    state.buffers[j] = None
            for pot in state.pots:
                if pot.order_id == order.id:
                    pot.order_id, pot.remaining = None, 0
            if state.joint.order_id == order.id:
                state.joint = JointOperation()
            events.append({"type": "order_expired", "order_id": order.id, "kind": order.kind})

    def potential(self, state=None):
        state = state or self.state
        total = 0.0
        for order in state.orders:
            if order.status != "active":
                continue
            stage = 0.0
            for item in [c.held for c in state.chefs] + state.buffers:
                if item is not None and item.order_id == order.id:
                    stage = max(stage, 1 if item.kind == "ingredient" else (5 if item.certified else 4))
            for pot in state.pots:
                if pot.order_id == order.id:
                    stage = max(stage, 2 if pot.remaining else 3)
            if state.joint.order_id == order.id:
                stage += state.joint.progress / order.sync_steps
            total += stage
        return total

    def observations(self):
        return tuple(self._encode([i] + [j for j in range(3) if j != i]) for i in range(3))

    def centralized_state(self):
        return self._encode([0, 1, 2])

    def _encode(self, order_players):
        s, c = self.state, self.config
        vector = [float((x, y) in self.floor) for y in range(7) for x in range(11)]
        facilities = [self.locations[k][0] for k in ("O", "T", "D", "S", "H")]
        facilities += self.pot_positions + self.buffer_positions
        for x, y in facilities:
            vector.extend((x / 10, y / 6))
        def encode_item(item):
            category = self.item_type(item, s)
            return [float(category == k) for k in range(6)] + [float(item.certified) if item else 0.0]
        for i in order_players:
            chef = s.chefs[i]
            vector.extend((float(i in self.active), chef.position[0] / 10, chef.position[1] / 6))
            vector.extend(float(chef.orientation == d) for d in DIRECTIONS)
            vector.extend(encode_item(chef.held))
        duration_scale = max(64, c.cook_steps + c.duration_jitter, c.sync_steps + c.duration_jitter)
        for pot in s.pots:
            order = self.order(pot.order_id, s)
            vector.extend((float(order is None), float(order is not None and order.kind == "normal"),
                           float(order is not None and order.kind == "joint"), pot.remaining / duration_scale))
        for item in s.buffers:
            vector.extend(encode_item(item))
        time_scale = max(c.horizon, c.normal_deadline, c.joint_deadline) + c.deadline_jitter
        for order in s.orders:
            vector.extend(float(order.kind == kind) for kind in ("normal", "joint"))
            vector.extend(float(order.status == status) for status in ("active", "completed", "expired"))
            vector.extend((float(order.stock), max(0, order.due_at - s.tick) / time_scale,
                           max(0, order.sync_due_at - s.tick) / time_scale,
                           order.cook_steps / duration_scale, order.sync_steps / duration_scale,
                           float(order.certified)))
        joint_order = self.order(s.joint.order_id, s)
        vector.extend((max(0, c.horizon - s.tick) / c.horizon, s.batch / c.max_batches,
                       s.joint.progress / joint_order.sync_steps if joint_order else 0.0))
        vector.extend(float(i in s.joint.pair) for i in order_players)
        return np.asarray(vector, dtype=np.float32)

    def _info(self, events, raw, shaping):
        return {"version": VERSION, "fingerprint": self.fingerprint,
                "tick": self.state.tick, "batch": self.state.batch,
                "score": self.state.score, "raw_reward": raw, "shaping_reward": shaping,
                "events": events, "metrics": copy.deepcopy(self.state.metrics)}

    def validate_state(self, state=None):
        state = state or self.state
        assert len(state.chefs) == 3 and len(state.pots) == 2 and len(state.buffers) == 2
        assert [o.kind for o in state.orders] == ["normal", "joint"]
        assert 0 <= state.tick <= self.config.horizon
        positions = [state.chefs[i].position for i in self.active]
        assert len(set(positions)) == len(positions)
        for i, chef in enumerate(state.chefs):
            assert chef.position in self.floor and chef.orientation in DIRECTIONS
            assert i in self.active or chef.held is None
        items = [c.held for c in state.chefs] + state.buffers
        for item in items:
            if item is None:
                continue
            assert item.kind in ("ingredient", "dish", "soup")
            if item.kind == "dish":
                assert item.order_id is None and not item.certified
            else:
                order = self.order(item.order_id, state)
                assert order is not None and order.status == "active"
                assert not item.certified or (item.kind == "soup" and order.kind == "joint" and order.certified)
        for pot in state.pots:
            assert pot.remaining >= 0
            assert pot.order_id is not None or pot.remaining == 0
            if pot.order_id is not None:
                assert self.order(pot.order_id, state).status == "active"
        for order in state.orders:
            assert order.status in ("active", "completed", "expired") and order.stock in (0, 1)
            assert order.created_at <= state.tick and order.due_at > order.created_at
            assert order.cook_steps > 0
            if order.kind == "joint":
                assert order.created_at < order.sync_due_at < order.due_at and order.sync_steps > 0
            else:
                assert not order.certified and order.sync_steps == 0 and order.sync_due_at == 0
            count = sum(item is not None and item.order_id == order.id for item in items)
            count += sum(p.order_id == order.id for p in state.pots)
            # A live ticket has exactly one ingredient, either in stock or in flight.
            assert order.stock + count == (1 if order.status == "active" else 0)
            if order.certified and order.status == "active":
                assert any(item is not None and item.order_id == order.id and item.certified for item in items)
        if state.joint.progress:
            order = self.order(state.joint.order_id, state)
            assert order is not None and order.status == "active"
            assert 0 < state.joint.progress < order.sync_steps
            assert len(state.joint.pair) == 2 and set(state.joint.pair).issubset(self.active)
            assert state.joint.donor in state.joint.pair
            donor = state.chefs[state.joint.donor]
            assert donor.held is not None and donor.held.kind == "soup"
            assert donor.held.order_id == order.id and not donor.held.certified
            assert state.tick <= order.sync_due_at
            for i in state.joint.pair:
                chef = state.chefs[i]
                assert adjacent(chef.position, chef.orientation) == self.handoff
                assert i == state.joint.donor or chef.held is None
        else:
            assert state.joint.order_id is None and state.joint.pair == () and state.joint.donor is None

    def render(self):
        """RGB array with the original two-dimensional Overcooked art."""
        from overcooked_ai_py.visualization.state_visualizer import StateVisualizer
        import pygame

        def render_item(item, position):
            if item is None:
                return None
            if item.kind == "dish":
                return SimpleNamespace(name="dish", position=position)
            order = self.order(item.order_id)
            ingredient = "onion" if order.kind == "joint" else "tomato"
            if item.kind == "ingredient":
                return SimpleNamespace(name=ingredient, position=position)
            return SimpleNamespace(name="soup", position=position, ingredients=[ingredient],
                                   is_idle=False, is_ready=True, _cooking_tick=order.cook_steps,
                                   cook_time=order.cook_steps)

        players = [SimpleNamespace(position=c.position, orientation=c.orientation,
                                   held_object=render_item(c.held, c.position))
                   for i, c in enumerate(self.state.chefs) if i in self.active]
        objects = {p: render_item(item, p) for p, item in zip(self.buffer_positions, self.state.buffers) if item}
        for p, pot in zip(self.pot_positions, self.state.pots):
            if pot.order_id is not None:
                order = self.order(pot.order_id)
                item = render_item(Item("soup", pot.order_id), p)
                item._cooking_tick = order.cook_steps - pot.remaining
                item.is_ready = pot.remaining == 0
                objects[p] = item
        grid = [["X" if t == "B" else t for t in row] for row in self.terrain]
        hud = {"score": self.state.score, "time_left": self.config.horizon - self.state.tick,
               "batch": self.state.batch,
               "normal_tomato": "{}, due in {}".format(
                   self.state.orders[0].status, max(0, self.state.orders[0].due_at - self.state.tick)),
               "joint_onion": "{}, due in {}, verified {}".format(
                   self.state.orders[1].status, max(0, self.state.orders[1].due_at - self.state.tick),
                   self.state.orders[1].certified),
               "cooperation": "{}/{}, window {}".format(
                   self.state.joint.progress, self.state.orders[1].sync_steps,
                   max(0, self.state.orders[1].sync_due_at - self.state.tick)),
               "food_stock": "tomato {} / onion {}".format(
                   self.state.orders[0].stock, self.state.orders[1].stock)}
        buffer_positions = self.buffer_positions

        class KitchenVisualizer(StateVisualizer):
            def _render_grid(self, surface, grid):
                super()._render_grid(surface, grid)
                for position in buffer_positions:
                    x, y = self._position_in_unscaled_pixels(position)
                    pygame.draw.rect(surface, (250, 200, 80), (x + 1, y + 1, 13, 13), 1)
                    pygame.draw.rect(surface, (250, 200, 80), (x + 5, y + 5, 5, 5), 1)

        colors = ["blue", "green", "red"]
        surface = KitchenVisualizer(
            tile_size=64, hud_font_size=14,
            player_colors=[colors[i] for i in range(3) if i in self.active]
        ).render_state(
            SimpleNamespace(players=players, objects=objects), grid, hud_data=hud
        )
        return pygame.surfarray.array3d(surface).transpose(1, 0, 2).copy()
