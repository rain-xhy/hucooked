from dataclasses import dataclass
from math import isfinite

NORTH, SOUTH, EAST, WEST, STAY, INTERACT = range(6)
WIDTH, HEIGHT = 13, 7
OFFSETS = (-WIDTH, WIDTH, 1, -1)
EMPTY, TOMATO, POTATO, EGG, BEEF, CUT_TOMATO, CUT_POTATO = range(7)
FIRST_MEAL = 7
RECIPE_NAMES = ("tomato_egg", "potato", "potato_tomato_beef")
BITS = {CUT_TOMATO: 1, CUT_POTATO: 2, EGG: 4, BEEF: 8}
RECIPES = (5, 2, 11)
DELIVERY_SIDE = (1, 0, 0)


def cell(x, y):
    return y * WIDTH + x


SOURCES = {TOMATO: cell(0, 2), POTATO: cell(0, 4), EGG: cell(12, 2), BEEF: cell(12, 4)}
BOARDS = tuple(cell(x, y) for x, y in ((2, 2), (4, 4), (10, 2), (8, 4)))
POTS = (cell(4, 2), cell(8, 2))
SERVERS = (cell(0, 3), cell(12, 3))
WALL = tuple(cell(6, y) for y in range(1, 6))
FLOORS = tuple(tuple(cell(x, y) for y in range(1, 6) for x in columns
                     if cell(x, y) not in BOARDS + POTS)
               for columns in (range(1, 6), range(7, 12)))
FLOOR = frozenset(FLOORS[0] + FLOORS[1])


@dataclass(frozen=True)
class PartitionConfig:
    layout: str = "sync"
    horizon: int = 300
    deadlines: tuple = (80, 60, 100)
    delivery_reward: float = 20.0

    def __post_init__(self):
        if self.layout not in ("sync", "counter"):
            raise ValueError("layout must be sync or counter")
        object.__setattr__(self, "deadlines", tuple(self.deadlines))
        if len(self.deadlines) != 3 or any(type(t) is not int or t <= 0 for t in self.deadlines):
            raise ValueError("Expected three positive integer deadlines")
        if type(self.horizon) is not int or self.horizon < 1:
            raise ValueError("horizon must be a positive integer")
        if not isfinite(self.delivery_reward) or self.delivery_reward < 0:
            raise ValueError("Invalid delivery reward")
