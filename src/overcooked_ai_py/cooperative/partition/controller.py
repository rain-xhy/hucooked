"""A reproducible, single-order-at-a-time baseline using the public native state."""
from collections import deque

from .model import (BEEF, BOARDS, EGG, FIRST_MEAL, FLOOR, INTERACT, OFFSETS, POTATO,
                    POTS, SERVERS, SOURCES, STAY, TOMATO, WALL, WIDTH, cell)


def route(state, chef, station):
    """Shortest path to any free interaction port, including the final turn."""
    start = int(state["pos"][chef])
    blocked = set(state["pos"]) - {start}
    queue = deque([(start, None)])
    visited = {start}
    while queue:
        pos, first = queue.popleft()
        for direction, offset in enumerate(OFFSETS):
            dest = pos + offset
            if dest == station:
                if first is not None:
                    return first
                return INTERACT if state["facing"][chef] == direction else direction
            if dest in FLOOR and dest not in visited and dest not in blocked:
                visited.add(dest)
                queue.append((dest, direction if first is None else first))
    return STAY


class ScriptedChefs:
    def __init__(self, env):
        self.env = env
        right = sorted((i for i, pos in enumerate(env.state["pos"]) if pos % WIDTH > 6),
                       key=lambda i: env.state["pos"][i])
        self.cook, self.courier = right
        self.left = next(i for i in range(3) if i not in right)
        self.jobs = deque()
        self.recipe = None

    def _plan(self, recipe):
        self.recipe = recipe
        left = self.left
        right = self.cook if recipe == 0 else self.courier
        target = int(recipe == 0)
        cook = right if target else left
        pot = POTS[target]
        wall = WALL[1 if recipe == 0 else 3]

        def work(chef, station, count=1):
            self.jobs.extend([((chef, station),)] * count)

        def pass_item(giver, receiver):
            if self.env.config.layout == "sync":
                self.jobs.append(((giver, wall), (receiver, wall)))
            else:
                work(giver, wall)
                work(receiver, wall)

        def vegetable(item):
            work(left, SOURCES[item])
            work(left, BOARDS[0 if item == TOMATO else 1], 4)
            if target:
                pass_item(left, right)
            work(cook, pot)

        if recipe in (1, 2):
            vegetable(POTATO)
        if recipe in (0, 2):
            vegetable(TOMATO)
        if recipe == 0:
            work(right, SOURCES[EGG])
            work(right, pot)
        if recipe == 2:
            work(right, SOURCES[BEEF])
            pass_item(right, left)
            work(left, pot)
        work(cook, pot, 5)
        work(cook, SERVERS[target])

    def actions(self):
        state = self.env.state
        if not self.jobs:
            self._plan(int(state["recipe"][0]))
        job = self.jobs[0]
        actions = [STAY] * 3
        for chef, station in job:
            actions[chef] = route(state, chef, station)
        # Clear unused chefs off ports or paths needed by the current operation.
        actors = {chef for chef, _ in job}
        for chef in (self.cook, self.courier):
            if chef in actors:
                continue
            parking = cell(11, 5) if chef == self.courier else cell(11, 1)
            if state["pos"][chef] != parking:
                blocked = {state["pos"][i] for i in range(3) if i != chef}
                queue = deque([(int(state["pos"][chef]), STAY)])
                seen = set(blocked)
                while queue:
                    pos, first = queue.popleft()
                    if pos in seen:
                        continue
                    seen.add(pos)
                    if pos == parking:
                        actions[chef] = first
                        break
                    for direction, offset in enumerate(OFFSETS):
                        if pos + offset in FLOOR:
                            queue.append((pos + offset, direction if first == STAY else first))
        ready = all(actions[chef] == INTERACT for chef, _ in job)
        if ready and len(self.jobs) == 1:
            chef, _ = job[0]
            ready = self.recipe in state["recipe"] and state["held"][chef] == FIRST_MEAL + self.recipe
        if ready:
            self.jobs.popleft()
        else:
            for chef, _ in job:
                if actions[chef] == INTERACT:
                    actions[chef] = STAY
        return actions
