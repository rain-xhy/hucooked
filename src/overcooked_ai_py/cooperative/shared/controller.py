from collections import deque
from .kitchen import DIRECTIONS, INTERACT, STAY
from .layouts import FLOOR, STATIONS


class ScriptedController:
    def __init__(self, env):
        if len(env.active) != 3:
            raise ValueError("The reference controller requires three active chefs")
        positions = {i: env.state.chefs[i].position for i in env.active}
        self.server = min(positions, key=lambda i: positions[i][0])
        others = [i for i in env.active if i != self.server]
        self.main = min(others, key=lambda i: abs(positions[i][0] - 4) + abs(positions[i][1] - 1))
        self.side = next(i for i in others if i != self.main)

    def actions(self, env):
        state = env.state
        plans = {}
        table = state.assembly
        cooker = state.cooker
        joint = cooker.item is not None and cooker.remaining == 0 and env.order(cooker.item.order_id).kind == "joint" and table.main is None
        for i in (self.server, self.main, self.side):
            chef = state.chefs[i]
            held = chef.held
            if i == self.server:
                if held is not None:
                    plan = ((1, 3), STATIONS["S"]) if held.kind == "meal" else ((2, 3), STATIONS["A"])
                elif table.plate is not None and table.plate.kind == "meal":
                    plan = ((2, 3), STATIONS["A"])
                elif table.main is not None and table.side is not None and table.plate is not None:
                    plan = ((2, 3), STATIONS["A"])
                elif joint:
                    plan = ((2, 2), STATIONS["G"])
                elif table.plate is None:
                    plan = ((1, 5), STATIONS["D"])
                else:
                    plan = ((2, 3), None)
            elif i == self.main:
                if held is not None:
                    plan = ((4, 2), STATIONS["G"]) if cooker.item is None else ((4, 1), STATIONS["M"])
                elif cooker.item is None and any(order.status == "active" for order in state.orders):
                    plan = ((4, 1), STATIONS["M"])
                elif cooker.item is not None and cooker.remaining == 0 and table.main is None:
                    plan = ((4, 2), STATIONS["G"])
                else:
                    plan = ((4, 2), None)
            else:
                if held is not None and held.kind == "raw_side":
                    plan = ((5, 2), STATIONS["K"])
                elif held is not None and held.kind == "chopped_side":
                    plan = ((4, 3), STATIONS["A"]) if table.side is None else ((5, 3), None)
                elif state.cutting.item is not None:
                    plan = ((5, 2), STATIONS["K"])
                elif table.side is None and any(order.status in ("active", "started") and order.assembled_at < 0 for order in state.orders):
                    plan = ((5, 1), STATIONS["L"])
                else:
                    plan = ((5, 3), None)
            plans[i] = plan
        actions = [STAY] * 3
        reserved = set()
        for i in (self.server, self.main, self.side):
            goal, station = plans[i]
            actions[i] = navigation_action(env, i, goal, station, reserved)
            if actions[i] < STAY:
                pos = state.chefs[i].position
                direction = DIRECTIONS[actions[i]]
                destination = (pos[0] + direction[0], pos[1] + direction[1])
                if destination in FLOOR:
                    reserved.add(destination)
        return actions


def navigation_action(env, chef_id, goal, station=None, reserved=()):
    chef = env.state.chefs[chef_id]
    if chef.position == goal:
        if station is None:
            return STAY
        facing = (station[0] - goal[0], station[1] - goal[1])
        return INTERACT if chef.orientation == facing else DIRECTIONS.index(facing)
    blocked = {env.state.chefs[i].position for i in env.active if i != chef_id} | set(reserved)
    queue = deque([(chef.position, None)])
    seen = {chef.position}
    while queue:
        position, first = queue.popleft()
        for action, direction in enumerate(DIRECTIONS):
            destination = (position[0] + direction[0], position[1] + direction[1])
            if destination not in FLOOR or destination in blocked or destination in seen:
                continue
            initial = action if first is None else first
            if destination == goal:
                return initial
            seen.add(destination)
            queue.append((destination, initial))
    return STAY
