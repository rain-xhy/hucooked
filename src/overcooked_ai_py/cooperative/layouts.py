"""One fixed task, three small geometry variants. B is a usable buffer."""

LAYOUTS = {
    "compact": (
        "XXPXXXXXPXX",
        "O  1   2  T",
        "X    B    X",
        "X    H    X",
        "X    B    X",
        "D    3    S",
        "XXXXXXXXXXX",
    ),
    "offset": (
        "XXXPXXXPXXX",
        "O  1   2  T",
        "X    B    X",
        "X    H    X",
        "X    B    X",
        "D    3    S",
        "XXXXXXXXXXX",
    ),
    "bottleneck": (
        "XXPXXXXXPXX",
        "O  1 X 2  T",
        "X    B    X",
        "X    H    X",
        "X    B    X",
        "D    3    S",
        "XXXXXXXXXXX",
    ),
}


def load_layout(name):
    rows = LAYOUTS[name]
    assert len(rows) == 7 and all(len(row) == 11 for row in rows)
    positions = {char: [] for char in "OPTDSHB123 "}
    terrain = []
    for y, row in enumerate(rows):
        line = []
        for x, char in enumerate(row):
            if char in positions:
                positions[char].append((x, y))
            line.append(" " if char in "123" else char)
        terrain.append(line)
    for char, count in (("O", 1), ("T", 1), ("P", 2), ("D", 1),
                        ("S", 1), ("H", 1), ("B", 2)):
        assert len(positions[char]) == count
    floor = {(x, y) for y, row in enumerate(terrain)
             for x, char in enumerate(row) if char == " "}
    for char in "123":
        assert len(positions[char]) == 1
        floor.add(positions[char][0])
    h = positions["H"][0]
    access = [(h[0] - 1, h[1]), (h[0] + 1, h[1])]
    assert all(p in floor for p in access)
    # Every walking tile and every facility must be reachable without crossing H.
    reached, stack = {next(iter(floor))}, [next(iter(floor))]
    while stack:
        x, y = stack.pop()
        for dx, dy in ((0, -1), (0, 1), (1, 0), (-1, 0)):
            neighbor = (x + dx, y + dy)
            if neighbor in floor and neighbor not in reached:
                reached.add(neighbor)
                stack.append(neighbor)
    assert reached == floor
    return terrain, positions, floor
