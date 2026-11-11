"""A public-state reference controller, not an RL or optimal-policy baseline.

It provides constructive task feasibility checks. Paths use only the current
map and player locations; no state writes, teleports, future orders or RNG reads.
"""

from collections import deque
from itertools import permutations

from .kitchen import DIRECTIONS, INTERACT, STAY, adjacent


class ReferenceController:
    def __init__(self):
        self.batch = None
        self.roles = {}
        self.helper_stage = 0

    def __call__(self, env):
        state = env.state
        if state.batch != self.batch:
            self.batch = state.batch
            active = list(env.active)
            if len(active) < 2:
                raise ValueError("This reference controller needs at least two active chefs")
            def cost(assignment):
                goals = [(1, 1), (9, 1), (1, 5)]
                return sum(abs(state.chefs[i].position[0] - goal[0]) +
                           abs(state.chefs[i].position[1] - goal[1])
                           for i, goal in zip(assignment, goals))
            assignment = min(permutations(active), key=cost)
            self.roles = {"joint": assignment[0], "normal": assignment[1]}
            if len(active) == 3:
                self.roles["helper"] = assignment[2]
            self.helper_stage = 0

        def go(i, position, direction, interact=True):
            chef = state.chefs[i]
            if chef.position == position:
                if chef.orientation != direction:
                    if not interact and adjacent(position, direction) in env.floor:
                        return STAY
                    return DIRECTIONS.index(direction)
                return INTERACT if interact else STAY
            occupied = {c.position for j, c in enumerate(state.chefs)
                        if j in env.active and j != i}
            queue, seen = deque([(chef.position, None)]), {chef.position}
            while queue:
                p, first = queue.popleft()
                if p == position:
                    return first
                for a, d in enumerate(DIRECTIONS):
                    nxt = adjacent(p, d)
                    if nxt in env.floor and nxt not in occupied and nxt not in seen:
                        seen.add(nxt)
                        queue.append((nxt, a if first is None else first))
            return STAY

        actions = [STAY, STAY, STAY]
        normal, joint = state.orders
        left_pot, right_pot = env.pot_positions
        helper = self.roles.get("helper")
        # A completed dish carrier always clears the serving position afterwards.
        for role in ("joint", "normal"):
            i = self.roles[role]
            chef = state.chefs[i]
            ticket = joint if role == "joint" else normal
            own_pot = left_pot if role == "joint" else right_pot
            if chef.held is not None and chef.held.kind == "soup":
                held_order = env.order(chef.held.order_id)
                if held_order.kind == "normal" or chef.held.certified:
                    actions[i] = go(i, (9, 5), (1, 0))
                else:
                    actions[i] = go(i, (4, 3), (1, 0))
            elif chef.held is not None and chef.held.kind == "ingredient":
                held_order = env.order(chef.held.order_id)
                p = left_pot if held_order.kind == "joint" else right_pot
                actions[i] = go(i, (p[0], 1), (0, -1))
            elif ticket.status == "active" and ticket.stock:
                if chef.held is None:
                    actions[i] = go(i, (1, 1) if role == "joint" else (9, 1),
                                    (-1, 0) if role == "joint" else (1, 0))
                else:
                    # Return a leftover plate so full buffers cannot cause a deadlock.
                    actions[i] = go(i, (1, 5), (-1, 0))
            elif ticket.status == "active" and any(p.order_id == ticket.id for p in state.pots):
                pot = next(p for p in state.pots if p.order_id == ticket.id)
                if chef.held is not None and chef.held.kind == "dish":
                    actions[i] = go(i, (own_pot[0], 1), (0, -1), pot.remaining == 0)
                elif role == "normal" and helper is not None:
                    # Take the supplied plate from the opposite side of B.
                    ready = state.buffers[0] is not None
                    actions[i] = go(i, (6, 2), (-1, 0), ready)
                else:
                    actions[i] = go(i, (1, 5), (-1, 0))
            elif role == "normal" and helper is None and joint.status == "active" and not joint.certified:
                actions[i] = go(i, (6, 3), (-1, 0))
            else:
                # Keep access to O, D, pots, H, and S unoccupied while idle.
                parking = (3, 3) if role == "joint" else (9, 3)
                actions[i] = go(i, parking, (-1, 0) if role == "joint" else (1, 0), False)

        if helper is not None:
            i = helper
            chef = state.chefs[i]
            if chef.held is not None and chef.held.kind == "soup":
                actions[i] = go(i, (9, 5), (1, 0))
            else:
                if self.helper_stage == 0:
                    if chef.held is not None and chef.held.kind == "dish":
                        self.helper_stage = 1
                    elif state.buffers[0] is not None or normal.status != "active":
                        self.helper_stage = 2
                if self.helper_stage == 1 and chef.held is None:
                    self.helper_stage = 2
                if self.helper_stage == 0:
                    actions[i] = go(i, (1, 5), (-1, 0))
                elif self.helper_stage == 1:
                    actions[i] = go(i, (4, 2), (1, 0), state.buffers[0] is None)
                elif joint.status == "active" and not joint.certified:
                    actions[i] = go(i, (6, 3), (-1, 0))
                else:
                    actions[i] = go(i, (7, 5), (0, 1), False)

        # Keep an operation going once the pair has committed to it.
        for i in state.joint.pair:
            actions[i] = INTERACT

        # A deterministic rotating reservation breaks equal-target conflicts.
        reserved = set()
        for i in sorted(env.active, key=lambda j: (j - state.tick) % 3):
            position = state.chefs[i].position
            target = adjacent(position, DIRECTIONS[actions[i]]) if actions[i] < 4 else position
            if target not in env.floor:
                target = position
            if target in reserved:
                actions[i] = STAY
            else:
                reserved.add(target)
        return tuple(actions)
