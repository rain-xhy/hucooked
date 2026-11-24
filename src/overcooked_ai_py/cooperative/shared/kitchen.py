import copy
from dataclasses import asdict, replace
from hashlib import sha256
from numbers import Integral
from pathlib import Path
import numpy as np
from .layouts import ASSEMBLY_PORTS, COOKER_PORTS, FLOOR, LAYOUT, SPAWNS, STATIONS
from .model import Assembly, Chef, Cooker, Cutting, Item, KitchenState, SharedKitchenConfig, Ticket


NORTH, SOUTH, EAST, WEST, STAY, INTERACT = range(6)
DIRECTIONS = ((0, -1), (0, 1), (1, 0), (-1, 0))
ITEM_KINDS = (None, "raw_main", "raw_side", "chopped_side", "cooked_main", "plate", "meal")
ORDER_STATUSES = ("queued", "active", "started", "served")
MIXED_ORDER_OBSERVATION_SLOTS = 8
COMPACT_ORDER_SNAPSHOT_FINGERPRINT = "55ee61adaeaf1e946b3f06380b05ce260e84b392f1add5cc5fd3f1e1f5cff0b4"
FIXED_ORDER_SNAPSHOT_FINGERPRINTS = (
    "a535f64e626888a63c33a07f6fccf723cee2133625d21badd09fccbbad7837ec",
    "4063de4b4a00b59041702d301c9d0faa5252f5b373e794268f0914afe51492d6",
)


def participants(*groups):
    return tuple(sorted(set(i for group in groups for i in group)))


class SharedKitchen:
    def __init__(self, config=None):
        self.config = config or SharedKitchenConfig()
        if not isinstance(self.config, SharedKitchenConfig):
            raise TypeError("config must be SharedKitchenConfig")
        self.active = self.config.active_players
        self.terrain = LAYOUT
        self.rng = np.random.default_rng()
        self.reset(seed=0)

    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        positions = list(SPAWNS)
        if self.config.shuffle_players:
            positions = [SPAWNS[int(i)] for i in self.rng.permutation(3)]
        chefs = [Chef(positions[i] if i in self.active else None) for i in range(3)]
        kinds = []
        if not self.config.continuous_orders:
            for index in range(max(self.config.normal_orders, self.config.joint_orders)):
                if index < self.config.normal_orders:
                    kinds.append("normal")
                if index < self.config.joint_orders:
                    kinds.append("joint")
        self.state = KitchenState(chefs, [Ticket(i, kind) for i, kind in enumerate(kinds)])
        self._release(self.state)
        self.validate_state()
        return self.observations(), self._info([], 0.0)

    def _release(self, state):
        count = sum(order.status in ("active", "started") for order in state.orders)
        if self.config.continuous_orders:
            for _ in range(self.config.active_limit - count):
                order_id = len(state.orders)
                kind = ("normal", "joint")[order_id % 2]
                state.orders.append(Ticket(order_id, kind, status="active", released_at=state.tick))
            return
        for order in state.orders:
            if count >= self.config.active_limit:
                break
            if order.status == "queued":
                order.status = "active"
                order.released_at = state.tick
                count += 1

    @property
    def terminated(self):
        return not self.config.continuous_orders and all(order.status == "served" for order in self.state.orders)

    @property
    def truncated(self):
        return self.state.tick >= self.config.horizon and not self.terminated

    def order(self, order_id):
        return self.state.orders[order_id]

    def step(self, actions):
        if self.terminated or self.truncated:
            raise RuntimeError("Reset before stepping a finished episode")
        try:
            actions = tuple(actions)
        except TypeError as error:
            raise ValueError("Expected three discrete actions") from error
        if len(actions) != 3 or any(isinstance(a, (bool, np.bool_)) or not isinstance(a, Integral) or not 0 <= a < 6 for a in actions):
            raise ValueError("Expected three integer actions in [0, 5]")
        if any(actions[i] != STAY for i in range(3) if i not in self.active):
            raise ValueError("Inactive chefs must stay")
        old = self.state
        state = copy.deepcopy(old)
        events = []
        requests = self._requests(old, actions, state)
        use_counts = {}
        for _, _, resources, _ in requests:
            for resource in resources:
                use_counts[resource] = use_counts.get(resource, 0) + 1
        conflicts = {key for key, value in use_counts.items() if value > 1}
        state.metrics["resource_conflicts"] += len(conflicts)
        for operation, actors, resources, data in sorted(requests, key=lambda r: (r[0], r[2])):
            if not conflicts.intersection(resources):
                self._apply(state, operation, actors, data, events)
        self._move(old, state, actions)
        if old.cooker.item is not None and old.cooker.remaining > 0 and state.cooker.item == old.cooker.item:
            state.cooker.remaining -= 1
            if state.cooker.remaining == 0:
                state.cooker.item = replace(state.cooker.item, kind="cooked_main")
                state.orders[state.cooker.item.order_id].cooked_at = old.tick + 1
                events.append({"type": "main_cooked", "order_id": state.cooker.item.order_id, "actors": [], "tick": old.tick + 1})
        state.tick += 1
        if state.tick < self.config.horizon:
            self._release(state)
        reward = state.score - old.score
        self.validate_state(state)
        self.state = state
        return self.observations(), reward, self.terminated, self.truncated, self._info(events, reward)

    def _requests(self, old, actions, state):
        result = []
        targets = {}
        for i in self.active:
            chef = old.chefs[i]
            if actions[i] == INTERACT:
                targets[i] = tuple(chef.position[j] + chef.orientation[j] for j in (0, 1))
        cooker_actors = tuple(i for i, target in targets.items() if target == STATIONS["G"] and old.chefs[i].position in COOKER_PORTS)
        if old.cooker.item is not None and old.cooker.remaining == 0 and old.assembly.main is None:
            ticket = old.orders[old.cooker.item.order_id]
            empty = tuple(i for i in cooker_actors if old.chefs[i].held is None)
            if ticket.kind == "joint":
                state.metrics["joint_ready_steps"] += 1
                if empty:
                    state.metrics["joint_attempt_steps"] += 1
                if len(empty) == 1:
                    state.metrics["unmatched_attempt_steps"] += 1
                if len(empty) == 2:
                    result.append(("unload", empty, ("assembly.main", "cooker"), None))
            elif empty:
                result.append(("unload", empty, ("assembly.main", "cooker"), None))
        for i, target in targets.items():
            chef = old.chefs[i]
            held = chef.held
            if target == STATIONS["M"]:
                if held is None:
                    result.append(("take_source", (i,), (), "raw_main"))
                elif held.kind == "raw_main":
                    result.append(("return_source", (i,), (), None))
            elif target == STATIONS["L"]:
                if held is None:
                    result.append(("take_source", (i,), (), "raw_side"))
                elif held.kind in ("raw_side", "chopped_side"):
                    result.append(("return_source", (i,), (), None))
            elif target == STATIONS["D"]:
                if held is None:
                    result.append(("take_source", (i,), (), "plate"))
                elif held.kind == "plate":
                    result.append(("return_source", (i,), (), None))
            elif target == STATIONS["G"] and chef.position in COOKER_PORTS:
                pending = next((order.id for order in old.orders if order.status == "active"), None)
                if old.cooker.item is None and held is not None and held.kind == "raw_main" and pending is not None:
                    result.append(("load", (i,), ("cooker",), pending))
            elif target == STATIONS["K"]:
                if old.cutting.item is None and held is not None and held.kind == "raw_side":
                    result.append(("place_cutting", (i,), ("cutting",), None))
                elif held is None and old.cutting.item is not None:
                    operation = "take_cutting" if old.cutting.item.kind == "chopped_side" else "chop"
                    result.append((operation, (i,), ("cutting",), None))
            elif target == STATIONS["B"]:
                if held is not None and old.buffer is None:
                    result.append(("place_buffer", (i,), ("buffer",), None))
                elif held is None and old.buffer is not None:
                    result.append(("take_buffer", (i,), ("buffer",), None))
            elif target == STATIONS["A"] and chef.position in ASSEMBLY_PORTS:
                table = old.assembly
                slot = {"chopped_side": "side", "plate": "plate", "meal": "plate"}.get(held.kind) if held else None
                if slot is not None and getattr(table, slot) is None:
                    result.append(("place_assembly", (i,), ("assembly." + slot,), slot))
                elif held is None and table.plate is not None and table.plate.kind == "meal":
                    result.append(("take_meal", (i,), ("assembly.plate",), None))
                elif held is None and table.main is not None and table.side is not None and table.plate is not None:
                    result.append(("assemble", (i,), ("assembly.main", "assembly.plate", "assembly.side"), None))
            elif target == STATIONS["S"] and held is not None and held.kind == "meal":
                result.append(("serve", (i,), ("order." + str(held.order_id),), None))
        return result

    def _apply(self, state, operation, actors, data, events):
        chef = state.chefs[actors[0]]
        event = {"type": operation, "actors": list(actors), "tick": state.tick + 1}
        if operation == "take_source":
            unit = len(state.unit_sources)
            source = {"raw_main": "main", "raw_side": "side", "plate": "plate"}[data]
            state.unit_sources.append(source)
            chef.held = Item(data, (unit,), contributors=actors)
            event["units"] = [unit]
        elif operation == "return_source":
            destination = state.wasted_units if chef.held.kind == "chopped_side" else state.returned_units
            destination.extend(chef.held.units)
            event["units"] = list(chef.held.units)
            chef.held = None
        elif operation == "load":
            state.cooker = Cooker(replace(chef.held, order_id=data, contributors=participants(chef.held.contributors, actors)), self.config.cook_steps)
            chef.held = None
            ticket = state.orders[data]
            ticket.status = "started"
            ticket.started_at = state.tick + 1
            ticket.main_loaded_by = actors
            event["order_id"] = data
        elif operation == "unload":
            value = state.cooker.item
            state.assembly.main = replace(value, contributors=participants(value.contributors, actors))
            state.cooker = Cooker()
            ticket = state.orders[value.order_id]
            ticket.unloaded_at = state.tick + 1
            ticket.unloaded_by = actors
            if ticket.kind == "joint":
                state.metrics["joint_successes"] += 1
            event["order_id"] = value.order_id
            event["units"] = list(value.units)
        elif operation == "place_cutting":
            state.cutting = Cutting(replace(chef.held, contributors=participants(chef.held.contributors, actors)))
            chef.held = None
        elif operation == "chop":
            value = state.cutting.item
            state.cutting.progress += 1
            state.cutting.item = replace(value, contributors=participants(value.contributors, actors), prepared_by=participants(value.prepared_by, actors))
            if state.cutting.progress == self.config.chop_steps:
                state.cutting.item = replace(state.cutting.item, kind="chopped_side")
            event["units"] = list(value.units)
            event["progress"] = state.cutting.progress
        elif operation == "take_cutting":
            chef.held = replace(state.cutting.item, contributors=participants(state.cutting.item.contributors, actors))
            state.cutting = Cutting()
        elif operation == "place_buffer":
            state.buffer = replace(chef.held, contributors=participants(chef.held.contributors, actors))
            chef.held = None
        elif operation == "take_buffer":
            chef.held = replace(state.buffer, contributors=participants(state.buffer.contributors, actors))
            state.buffer = None
        elif operation == "place_assembly":
            setattr(state.assembly, data, replace(chef.held, contributors=participants(chef.held.contributors, actors)))
            chef.held = None
            event["slot"] = data
        elif operation == "assemble":
            table = state.assembly
            main, side, plate = table.main, table.side, table.plate
            joined = participants(main.contributors, side.contributors, plate.contributors, actors)
            meal = Item("meal", main.units + side.units + plate.units, main.order_id, joined, side.prepared_by)
            ticket = state.orders[main.order_id]
            ticket.assembled_at = state.tick + 1
            ticket.side_prepared_by = side.prepared_by
            ticket.plate_supplied_by = plate.contributors
            ticket.assembled_by = actors
            state.assembly = Assembly(plate=meal)
            event["order_id"] = main.order_id
            event["units"] = list(meal.units)
        elif operation == "take_meal":
            chef.held = replace(state.assembly.plate, contributors=participants(state.assembly.plate.contributors, actors))
            state.assembly.plate = None
            event["order_id"] = chef.held.order_id
        elif operation == "serve":
            meal = chef.held
            ticket = state.orders[meal.order_id]
            if ticket.status != "started":
                raise ValueError("Only an unfinished order can be delivered")
            ticket.status = "served"
            ticket.completed_at = state.tick + 1
            ticket.served_by = actors
            ticket.contributors = participants(meal.contributors, actors)
            ticket.delivered_units = meal.units
            state.score += getattr(self.config, ticket.kind + "_reward")
            state.metrics[ticket.kind + "_completed"] += 1
            chef.held = None
            event["order_id"] = meal.order_id
            event["units"] = list(meal.units)
        events.append(event)

    def _move(self, old, state, actions):
        desired = {}
        for i in self.active:
            position = old.chefs[i].position
            if actions[i] < STAY:
                direction = DIRECTIONS[actions[i]]
                state.chefs[i].orientation = direction
                destination = tuple(position[j] + direction[j] for j in (0, 1))
                desired[i] = destination if destination in FLOOR else position
            else:
                desired[i] = position
        blocked = set()
        for i in self.active:
            for j in self.active:
                if i >= j:
                    continue
                if desired[i] == desired[j] or (desired[i] == old.chefs[j].position and desired[j] == old.chefs[i].position):
                    blocked.update((i, j))
        changed = True
        while changed:
            changed = False
            for i in self.active:
                if i in blocked:
                    continue
                if any(desired[i] == old.chefs[j].position and (j in blocked or desired[j] == old.chefs[j].position) for j in self.active if j != i):
                    blocked.add(i)
                    changed = True
        state.metrics["movement_conflicts"] += sum(i in blocked and desired[i] != old.chefs[i].position for i in self.active)
        for i in self.active:
            state.chefs[i].position = old.chefs[i].position if i in blocked else desired[i]

    def episode_metrics(self):
        state = self.state
        count = state.metrics["normal_completed"] + state.metrics["joint_completed"]
        metrics = dict(state.metrics)
        metrics.update({
            "team_score": state.score,
            "meals_completed": count,
            "ticks": state.tick,
            "joint_attempt_success_rate": state.metrics["joint_successes"] / state.metrics["joint_attempt_steps"] if state.metrics["joint_attempt_steps"] else 0.0,
            "joint_wait_steps": state.metrics["joint_ready_steps"] - state.metrics["joint_successes"],
            "three_contributor_meals": sum(len(order.contributors) == 3 for order in state.orders if order.status == "served"),
        })
        if self.config.continuous_orders:
            metrics["meals_per_100_steps"] = 100.0 * count / max(state.tick, 1)
        else:
            metrics["orders_total"] = len(state.orders)
            metrics["completion_rate"] = count / len(state.orders)
        return metrics

    def _info(self, events, reward):
        return {"events": events, "metrics": self.episode_metrics(), "team_reward": reward, "orders": [asdict(order) for order in self.state.orders]}

    def _item_features(self, item, order_slots, slot_count):
        kind = None if item is None else item.kind
        slot = -1 if item is None else order_slots.get(item.order_id, -1)
        return [float(kind == candidate) for candidate in ITEM_KINDS] + [float(slot == i) for i in range(-1, slot_count)]

    def _observed_orders(self, state):
        orders = state.orders
        if not self.config.continuous_orders:
            return orders
        slot_count = max(MIXED_ORDER_OBSERVATION_SLOTS, self.config.active_limit)
        slots = orders[:slot_count]
        for order_id in range(len(slots), slot_count):
            slots.append(Ticket(order_id, ("normal", "joint")[order_id % 2]))
        for order in orders[slot_count:]:
            reusable = (index for index, ticket in enumerate(slots) if 0 <= ticket.completed_at <= order.released_at)
            index = min(reusable, key=lambda index: slots[index].id)
            slots[index] = order
        return slots

    def centralized_state(self):
        state = self.state
        orders = self._observed_orders(state)
        slot_count = len(orders)
        order_slots = {order.id: slot for slot, order in enumerate(orders)}
        values = [state.tick / self.config.horizon]
        for row in LAYOUT:
            for tile in row:
                values.extend(float(tile == candidate) for candidate in "X MLKGABDS")
        for i, chef in enumerate(state.chefs):
            values.append(float(i in self.active))
            values.extend(float(chef.position == (x, y)) for y in range(7) for x in range(7))
            values.extend(float(chef.orientation == direction) for direction in DIRECTIONS)
            values.extend(self._item_features(chef.held, order_slots, slot_count))
        for item in (state.cooker.item, state.cutting.item, state.assembly.main, state.assembly.side, state.assembly.plate, state.buffer):
            values.extend(self._item_features(item, order_slots, slot_count))
        values.extend((state.cooker.remaining / self.config.cook_steps, state.cutting.progress / self.config.chop_steps))
        for order in orders:
            values.extend(float(order.status == status) for status in ORDER_STATUSES)
            values.extend(float(order.kind == kind) for kind in ("normal", "joint"))
            values.extend((max(order.released_at, 0) / self.config.horizon, max(order.started_at, 0) / self.config.horizon))
        return np.asarray(values, dtype=np.float32)

    def observations(self):
        common = self.centralized_state()
        return [np.concatenate((common, np.eye(3, dtype=np.float32)[i])) for i in range(3)]

    def snapshot(self):
        return {"version": 1, "fingerprint": self.fingerprint(), "config": asdict(self.config), "state": asdict(self.state), "rng": copy.deepcopy(self.rng.bit_generator.state)}

    @staticmethod
    def fingerprint():
        root = Path(__file__).parent
        sources = ((root / name).read_bytes().replace(b"\r\n", b"\n") for name in ("model.py", "layouts.py", "kitchen.py"))
        return sha256(b"".join(sources)).hexdigest()

    def restore(self, snapshot):
        fingerprint = snapshot.get("fingerprint")
        previous_fixed = not self.config.continuous_orders and fingerprint in FIXED_ORDER_SNAPSHOT_FINGERPRINTS
        if snapshot.get("version") != 1 or (not previous_fixed and fingerprint not in (COMPACT_ORDER_SNAPSHOT_FINGERPRINT, self.fingerprint())):
            raise ValueError("Incompatible snapshot")
        if SharedKitchenConfig(**snapshot["config"]) != self.config:
            raise ValueError("Snapshot configuration does not match")
        data = copy.deepcopy(snapshot["state"])
        def item(value):
            if value is None:
                return None
            for key in ("units", "contributors", "prepared_by"):
                value[key] = tuple(value[key])
            return Item(**value)
        data["chefs"] = [Chef(None if chef["position"] is None else tuple(chef["position"]), tuple(chef["orientation"]), item(chef["held"])) for chef in data["chefs"]]
        for value in data["orders"]:
            for key in ("main_loaded_by", "unloaded_by", "side_prepared_by", "plate_supplied_by", "assembled_by", "served_by", "contributors", "delivered_units"):
                value[key] = tuple(value[key])
        data["orders"] = [Ticket(**value) for value in data["orders"]]
        data["cooker"] = Cooker(item(data["cooker"]["item"]), data["cooker"]["remaining"])
        data["cutting"] = Cutting(item(data["cutting"]["item"]), data["cutting"]["progress"])
        data["assembly"] = Assembly(**{key: item(value) for key, value in data["assembly"].items()})
        data["buffer"] = item(data["buffer"])
        candidate = KitchenState(**data)
        self.validate_state(candidate)
        rng = np.random.default_rng()
        rng.bit_generator.state = copy.deepcopy(snapshot["rng"])
        self.state = candidate
        self.rng = rng

    def validate_state(self, state=None):
        state = self.state if state is None else state
        def require(condition, message):
            if not condition:
                raise ValueError(message)
        require(type(state.tick) is int and 0 <= state.tick <= self.config.horizon, "Invalid episode time")
        require(len(state.chefs) == 3, "Expected three chef slots")
        positions = [state.chefs[i].position for i in self.active]
        require(len(set(positions)) == len(positions) and all(position in FLOOR for position in positions), "Invalid chef positions")
        for i, chef in enumerate(state.chefs):
            require(chef.orientation in DIRECTIONS, "Invalid orientation")
            if i not in self.active:
                require(chef.position is None and chef.held is None, "Inactive chef has state")
        order_count = len(state.orders)
        if self.config.continuous_orders:
            require(all(order.status != "queued" and order.kind == ("normal", "joint")[index % 2]
                        for index, order in enumerate(state.orders)), "Invalid continuous order sequence")
        else:
            require(order_count == self.config.normal_orders + self.config.joint_orders, "Invalid order count")
            require(sum(order.kind == "normal" for order in state.orders) == self.config.normal_orders, "Invalid recipe counts")
        live = [chef.held for chef in state.chefs] + [state.cooker.item, state.cutting.item, state.assembly.main, state.assembly.side, state.assembly.plate, state.buffer]
        units = []
        for value in live:
            if value is None:
                continue
            require(value.kind in ITEM_KINDS[1:], "Invalid item kind")
            sources = {"raw_main": ("main",), "cooked_main": ("main",), "raw_side": ("side",), "chopped_side": ("side",), "plate": ("plate",), "meal": ("main", "side", "plate")}[value.kind]
            require(len(value.units) == len(sources), "Invalid item size")
            for unit, source in zip(value.units, sources):
                require(type(unit) is int and 0 <= unit < len(state.unit_sources) and state.unit_sources[unit] == source, "Invalid material lineage")
            require(all(i in self.active for i in value.contributors + value.prepared_by), "Invalid contributors")
            require(tuple(sorted(set(value.contributors))) == value.contributors and tuple(sorted(set(value.prepared_by))) == value.prepared_by, "Duplicate contributors")
            if value.kind in ("cooked_main", "meal") or value is state.cooker.item:
                require(0 <= value.order_id < order_count and state.orders[value.order_id].status == "started", "Invalid main order")
            else:
                require(value.order_id == -1, "Unexpected order association")
            units.extend(value.units)
        cooker = state.cooker
        require(type(cooker.remaining) is int and 0 <= cooker.remaining <= self.config.cook_steps, "Invalid cooking timer")
        require((cooker.item is None and cooker.remaining == 0) or (cooker.item is not None and cooker.item.kind == ("raw_main" if cooker.remaining else "cooked_main")), "Invalid cooker state")
        cutting = state.cutting
        require(type(cutting.progress) is int and 0 <= cutting.progress <= self.config.chop_steps, "Invalid cutting progress")
        require((cutting.item is None and cutting.progress == 0) or (cutting.item is not None and cutting.item.kind == ("chopped_side" if cutting.progress == self.config.chop_steps else "raw_side")), "Invalid cutting state")
        table = state.assembly
        for value, allowed in ((table.main, ("cooked_main",)), (table.side, ("chopped_side",)), (table.plate, ("plate", "meal"))):
            require(value is None or value.kind in allowed, "Invalid assembly slot")
        active_count = sum(order.status in ("active", "started") for order in state.orders)
        require(active_count <= self.config.active_limit, "Too many active orders")
        if self.config.continuous_orders and state.tick < self.config.horizon:
            require(active_count == self.config.active_limit, "Missing continuous orders")
        for index, order in enumerate(state.orders):
            require(order.id == index and order.kind in ("normal", "joint") and order.status in ORDER_STATUSES, "Invalid order")
            mains = [value for value in live if value is not None and value.order_id == index]
            require(len(mains) == (1 if order.status == "started" else 0), "Order main is missing or duplicated")
            times = [order.released_at, order.started_at, order.cooked_at, order.unloaded_at, order.assembled_at, order.completed_at]
            require(all(type(t) is int and -1 <= t <= state.tick for t in times), "Invalid order times")
            occurred = [t for t in times if t >= 0]
            require(occurred == sorted(occurred), "Order times are out of sequence")
            if order.status == "served":
                require(len(order.delivered_units) == 3 and all(t >= 0 for t in times), "Incomplete delivery record")
                require(len(order.unloaded_by) == 2 if order.kind == "joint" else len(order.unloaded_by) >= 1, "Invalid unloading record")
                require(all(i in self.active for i in order.contributors), "Invalid delivered contributors")
                require(tuple(state.unit_sources[u] for u in order.delivered_units) == ("main", "side", "plate"), "Invalid delivered materials")
            else:
                require(not order.delivered_units and order.completed_at == -1, "Unexpected delivery")
            units.extend(order.delivered_units)
        units.extend(state.returned_units)
        units.extend(state.wasted_units)
        require(len(units) == len(state.unit_sources) and set(units) == set(range(len(state.unit_sources))), "Materials were lost or duplicated")
        score = sum(getattr(self.config, order.kind + "_reward") for order in state.orders if order.status == "served")
        require(np.isfinite(state.score) and abs(state.score - score) < 1e-7, "Invalid team score")
        for kind in ("normal", "joint"):
            require(state.metrics[kind + "_completed"] == sum(order.kind == kind and order.status == "served" for order in state.orders), "Invalid completion counter")
        require(state.metrics["joint_successes"] == sum(order.kind == "joint" and order.unloaded_at >= 0 for order in state.orders), "Invalid joint counter")
        return True

    def render(self):
        from .rendering import render
        return render(self)
