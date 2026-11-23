LAYOUT = (
    "XXXXMXX",
    "X     L",
    "X  G  K",
    "S  A  X",
    "X  B  X",
    "D     X",
    "XXXXXXX",
)

STATIONS = {c: (x, y) for y, row in enumerate(LAYOUT) for x, c in enumerate(row) if c not in "X "}
FLOOR = frozenset((x, y) for y, row in enumerate(LAYOUT) for x, c in enumerate(row) if c == " ")
COOKER_PORTS = frozenset(((2, 2), (4, 2)))
ASSEMBLY_PORTS = frozenset(((2, 3), (4, 3)))
SPAWNS = ((2, 3), (4, 2), (5, 2))
