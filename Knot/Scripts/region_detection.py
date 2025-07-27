from pathNode import Node, KnotManager
from collections import deque
import colorsys
from shapely.geometry import LineString
# Manager instance to replace all global state
knot_manager = KnotManager()

from pathNode import TurningPoint, Node, KnotPoint

# Keep this class — no change needed yet
class Section:
    def __init__(self, section_id, start, end, over_under, crossings):
        self.id = section_id
        self.start = start
        self.end = end
        self.over_under = over_under  # 1: overpass, -1: underpass, 0: none
        self.crossings = crossings

    def __repr__(self):
        kind = (
            "Overpass"
            if self.over_under == 1
            else "Underpass" if self.over_under == -1 else "Flat"
        )
        return f"Section {self.id}: {self.start} -> {self.end} [{kind}], crosses: {list(self.crossings)}"


class TurningPoint:
    def __init__(self, point, is_agent=False):
        self.point = point
        self.is_agent = is_agent

    def __repr__(self):
        return f"{'Agent' if self.is_agent else 'Turn'}@{self.point}"


def reset_globals():
    knot_manager.reset()


def read_path():
    rows = int(input("Enter the number of rows: "))
    cols = int(input("Enter the number of columns: "))
    matrixA = []
    print("Enter the matrix row by row (comma separated):")
    for _ in range(rows):
        row = list(map(int, input().split(",")))
        matrixA.append(row)

    while True:
        print("Enter the path entry point (row, col):")
        entry_input = input().split(",")
        if len(entry_input) == 2:
            entryPoint = tuple(map(int, entry_input))
            if (
                0 <= entryPoint[0] < rows
                and 0 <= entryPoint[1] < cols
                and matrixA[entryPoint[0]][entryPoint[1]] in (-1, 1)
            ):
                break

    while True:
        print("Enter the path exit point (row, col):")
        exit_input = input().split(",")
        if len(exit_input) == 2:
            exitPoint = tuple(map(int, exit_input))
            if (
                0 <= exitPoint[0] < rows
                and 0 <= exitPoint[1] < cols
                and matrixA[exitPoint[0]][exitPoint[1]] in (-1, 1)
            ):
                break

    knot_manager.set_matrix(matrixA, entryPoint, exitPoint)
    return matrixA, entryPoint, exitPoint


def determine_crossing_behavior(start, end, matrixA):
    r1, c1 = start
    r2, c2 = end
    crossing_cells = set()
    over = False
    under = False

    if r1 == r2:  # Horizontal
        for c in range(min(c1, c2) + 1, max(c1, c2)):
            if matrixA[r1][c] == 3:
                crossing_cells.add((r1, c))
                if matrixA[r1][c1] == 1:
                    under = True
                elif matrixA[r1][c1] == -1:
                    over = True
    elif c1 == c2:  # Vertical
        for r in range(min(r1, r2) + 1, max(r1, r2)):
            if matrixA[r][c1] == 3:
                crossing_cells.add((r, c1))
                if matrixA[r1][c1] == 1:
                    under = True
                elif matrixA[r1][c1] == -1:
                    over = True

    if over and not under:
        return 1, crossing_cells
    if under and not over:
        return -1, crossing_cells
    if over and under:
        print(f"⚠️ Conflicting crossing behavior: {start}->{end}")
    return 0, crossing_cells


def search_next_turn(matrixA, row, col, direction):
    rows, cols = len(matrixA), len(matrixA[0])
    if direction == "row":
        for c in range(col - 1, -1, -1):
            if matrixA[row][c] in {1, -1}:
                return (row, c)
        for c in range(col + 1, cols):
            if matrixA[row][c] in {1, -1}:
                return (row, c)
    else:
        for r in range(row - 1, -1, -1):
            if matrixA[r][col] in {1, -1}:
                return (r, col)
        for r in range(row + 1, rows):
            if matrixA[r][col] in {1, -1}:
                return (r, col)
    return None


def find_starting_direction(matrixA, row, col):
    if search_next_turn(matrixA, row, col, "row"):
        return "row"
    if search_next_turn(matrixA, row, col, "col"):
        return "col"
    return None


def detect_loop(path_list, path_set, current_point):
    if current_point in path_set:
        i = path_list.index(current_point)
        return path_list[i:]
    return None


def detect_pocket(path_list, matrixA):
    """
    Look for pattern: crossing → turn → crossing that leaves exactly one 1/-1 cell
    between the two strands.  If found, return the minimal cycle [A,B,C].
    """
    if len(path_list) < 3:
        return None

    a, b, c = path_list[-3], path_list[-2], path_list[-1]
    r1, c1 = a
    r2, c2 = b
    r3, c3 = c

    # 1) both endpoints on crossings
    if matrixA[r1][c1] != 3 or matrixA[r3][c3] != 3:
        return None

    # 2) b must be a turn
    if not ((r1 == r2 == r3) or (c1 == c2 == c3)):  # i.e. not straight
        # 3) In a right‑angle pocket, the “cell in the pocket” is at (r1, c3) or (r3, c1)
        #    depending on whether you turned vertical→horizontal or vice versa.
        pocket = (r1, c3) if (r1 != r3) else (r3, c1)
        pr, pc = pocket
        if (
            0 <= pr < len(matrixA)
            and 0 <= pc < len(matrixA[0])
            and matrixA[pr][pc] in {1, -1}
        ):
            # we’ve found a “pocket” – return the triple as a mini‑loop
            return [a, b, c]
    return None

def find_enclosed_area(matrix, loop_boundary):
    rows, cols = len(matrix), len(matrix[0])
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    # Treat loop as a wall
    loop_set = set(loop_boundary)

    # Step 1: Flood fill from outer borders, but don't pass through loop boundary
    exterior = set()
    queue = deque()

    for r in range(rows):
        for c in (0, cols - 1):
            queue.append((r, c))
        for c in range(cols):
            for rr in (0, rows - 1):
                queue.append((rr, c))

    visited = set()
    while queue:
        r, c = queue.popleft()
        if (
            (r, c) in visited
            or (r, c) in loop_set
            or not (0 <= r < rows and 0 <= c < cols)
        ):
            continue
        visited.add((r, c))
        exterior.add((r, c))
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) not in loop_set:
                queue.append((nr, nc))

    # Step 2: Everything not visited and not in loop is interior
    enclosed = set()
    print(f"\n🔒 Interior Region Detection (Enclosed Area Check):")
    for r in range(rows):
        for c in range(cols):
            if (r, c) not in exterior and (r, c) not in loop_set:
                enclosed.add((r, c))
                print(f"  🧩 ({r}, {c}) → {matrix[r][c]}")

    # Step 3: Extract meaningful values (1 or -1)
    value_cells = {(r, c) for (r, c) in enclosed if matrix[r][c] in {1, -1}}

    # Visual debug of grid
    print("\n🖼️ Matrix View:")
    for r in range(rows):
        line = ""
        for c in range(cols):
            pos = (r, c)
            if pos in loop_set:
                line += " #"
            elif pos in value_cells:
                line += " *"
            elif pos in enclosed:
                line += " ."
            else:
                line += "  "
        print(line)

    return enclosed, bool(value_cells), value_cells


def handle_loop(
        loop,
        processed_regions,
        entryPoint,
        exitPoint,
        path_list,
        loop_id,
        current_agents,
        walked_turning_points,
        cell_info,
        force_agent_if_enclosed=False,
):
    path_set = set(path_list)
    print("Handle loop here")

    enclosed_area, contains_value, cells_with_values = find_enclosed_area(
        knot_manager.matrix, loop
    )
    if not contains_value:
        return set()

    if force_agent_if_enclosed:
        print(f"  ⚠️ Forcing agent assignment for loop {loop_id} due to enclosed 1/-1")

        agent_points = []
        for p in loop:
            val = knot_manager.matrix[p[0]][p[1]]
            if val in {1, -1} and p != entryPoint and p != exitPoint:
                agent_points.append(p)
                print(f"    ➕ Candidate agent point: {p} (value={val})")

        if not agent_points:
            print(f"  ⚠️ No valid agent points assigned for loop {loop_id}")
            return set()

        knot_manager.loop_registry[loop_id] = {"path": loop, "agents": []}
        new_agents = set()

        for p in agent_points:
            # Check if agent already exists
            found = None
            for aid, pt_obj in knot_manager.agent_registry.items():
                if pt_obj.pos_2d() == p:
                    found = aid
                    break

            if found is not None:
                print(f"  ℹ️ [Forced] Agent already exists at {p}, reusing Agent {found}")
                aid = found
            else:
                aid = knot_manager.register_agent(knot_manager.graph.add_point(*p, is_agent=True))

                print(f"  ✅ [Forced] Assigned NEW Agent {aid} at {p}")
                new_agents.add(p)

            knot_manager.loop_registry[loop_id]["agents"].append((aid, p))

        return new_agents

    # 2.  Per‑cell bookkeeping (normal path) ------------------------------------------
    max_cells = len(cells_with_values)
    for c in cells_with_values:
        if c not in cell_info:
            cell_info[c] = {"done": 0, "max": max_cells}
            print(f"    init {c}: 0/{max_cells}")

    # 3.  Determine uncovered cells & verbose evaluation -----------------
    uncovered_cells = []
    for c in cells_with_values:
        info = cell_info[c]
        state = f"{info['done']}/{info['max']}"
        if info["done"] < info["max"] and c not in path_set:
            uncovered_cells.append(c)
            note = "needs-agent"
        elif c in path_set:
            note = "visited"
        else:
            note = "covered"
        print(f"    eval {c}: {state} {note}")

    if not uncovered_cells:
        return set()

    # 4.  Standard processed‑area bookkeeping -------------------------------
    proc_set = (
        processed_regions if isinstance(processed_regions, set) else set(processed_regions)
    )
    proc_set.update(enclosed_area)
    proc_set.update(cells_with_values)
    proc_set.update(loop)
    if isinstance(processed_regions, set):
        processed_regions.clear()
        processed_regions.update(proc_set)

    # 5.  Turning‑point reuse guard -----------------------------------------
    def is_turning_point(prev, curr, nxt):
        return (prev[0] == curr[0] and nxt[0] != curr[0]) or (
                prev[1] == curr[1] and nxt[1] != curr[1]
        )

    turning_points = {
        loop[i]
        for i in range(1, len(loop) - 1)
        if is_turning_point(loop[i - 1], loop[i], loop[i + 1])
    }
    reused = turning_points & walked_turning_points
    walked_turning_points.update(turning_points)
    if reused:
        return set()

    # 6.  Agent candidate selection (corner heuristic) ----------------------
    def select_priority_agents(loop):
        pts = [p for p in loop if knot_manager.matrix[p[0]][p[1]] in {1, -1}]
        if len(pts) < 4:
            return pts
        top = min(pts, key=lambda p: p[0])
        bottom = max(pts, key=lambda p: p[0])
        left = min(pts, key=lambda p: p[1])
        right = max(pts, key=lambda p: p[1])
        return list({top, bottom, left, right})

    agent_points = select_priority_agents(loop)

    # Fallback: fill in up to 4 if needed
    def find_best_agent_point(start_idx, path):
        n = len(path)
        for off in range(n):
            p0 = path[(start_idx + off - 1) % n]
            p1 = path[(start_idx + off) % n]
            p2 = path[(start_idx + off + 1) % n]
            v = knot_manager.matrix[p1[0]][p1[1]]
            if is_turning_point(p0, p1, p2) and v in {1, -1}:
                return p1
        for off in range(n):
            p = path[(start_idx + off) % n]
            if knot_manager.matrix[p[0]][p[1]] in {1, -1}:
                return p
        return None

    used = set(agent_points)
    i = 0
    while len(agent_points) < 4 and i < len(loop):
        pt = find_best_agent_point(i, loop)
        if pt and pt not in used:
            agent_points.append(pt)
            used.add(pt)
        i += 1

    new_agents = [
        pt for pt in agent_points if pt not in current_agents and pt != entryPoint and pt != exitPoint
    ]
    if not new_agents:
        return set()

    knot_manager.loop_registry[loop_id] = {"path": loop, "agents": []}
    for p in new_agents:
        aid = knot_manager.register_agent(knot_manager.graph.add_point(*p))
        knot_manager.loop_registry[loop_id]["agents"].append((aid, p))
        print(f"  ✅ Assigned Agent {aid} at {p}")

    # 8.  Increment progress & print updates ----------------------------
    for c in uncovered_cells:
        cell_info[c]["done"] += 1
        info = cell_info[c]
        print(f"    update {c}: {info['done']}/{info['max']}")

    min_done = min(cell_info[c]["done"] for c in cells_with_values)
    loop_max = max_cells
    print(f"    progress {min_done}/{loop_max}")
    if min_done >= loop_max:
        print(f"  ✅ Loop {loop_id} fully covered!")

    return set(new_agents)


def trace_knot_path(matrixA, entryPoint, exitPoint):
    def is_opposite_corners(p1, p2, loop):
        xs = [x for x, _ in loop]
        ys = [y for _, y in loop]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        corners = [
            ((min_x, min_y), (max_x, max_y)),
            ((min_x, max_y), (max_x, min_y)),
        ]
        return (p1, p2) in corners or (p2, p1) in corners

    cell_coverage = {}
    walked_turning_points = set()
    knot_manager.reset()
    knot_manager.set_matrix(matrixA, entryPoint, exitPoint)
    processed_counts = {}
    seen_loops = set()
    crossing_events = []  # (point, old_idx, new_idx)
    loop_id = 1
    section_id = 1
    sections = []
    all_agents = set()

    # Seed agent
    a0 = knot_manager.graph.add_point(*entryPoint, is_agent=True)
    knot_manager.register_agent(a0)

    direction = find_starting_direction(matrixA, *entryPoint)
    if not direction:
        return [], set(), {}, []

    prev_turn, prev_dir = entryPoint, direction
    section_buf = []
    current = entryPoint
    path_list = [current]
    path_set = {current}

    while current != exitPoint:
        nxt = search_next_turn(matrixA, *current, direction)
        if not nxt:
            break

        r1, c1 = current
        r2, c2 = nxt
        if direction == "row":
            step = 1 if c2 > c1 else -1
            pts = [(r1, c) for c in range(c1 + step, c2 + step, step)]
        else:
            step = 1 if r2 > r1 else -1
            pts = [(r, c1) for r in range(r1 + step, r2 + step, step)]

        print(f"Tracing segment from {current} to {nxt}: {pts}")

        for p in pts:
            section_buf.append(p)

            if p in path_set:
                print("\n--- Loop detection triggered ---")
                print(f"  Current point: {p}")
                print(f"  Full search path ({len(path_list)} pts): {path_list}")

                old_idx = path_list.index(p)
                new_idx = len(path_list)
                crossing_events.append((p, old_idx, new_idx))

                # Same-point loop
                loop1 = path_list[old_idx:]
                if tuple(loop1) not in seen_loops:
                    print(f"\n🔁 Loop {loop_id}: same-point created by {p}")
                    print(f"  Loop boundary points: {loop1}")
                    enclosed1, has1, cells1 = find_enclosed_area(matrixA, loop1)
                    if has1:
                        print("  Enclosed cells:")
                        for cell in sorted(cells1):
                            val = matrixA[cell[0]][cell[1]]
                            status = "visited" if cell in path_set else "new"
                            print(f"    {cell}: value={val}, {status}")
                    else:
                        print("  No enclosed 1/-1 cells.")

                    seen_loops.add(tuple(loop1))
                    nas = handle_loop(
                        loop1,
                        processed_counts,
                        entryPoint,
                        exitPoint,
                        path_list,
                        loop_id,
                        all_agents,
                        walked_turning_points,
                        cell_coverage,
                    )
                    if nas:
                        all_agents.update(nas)
                        loop_id += 1

                # Between-crossings loop
                if len(crossing_events) >= 2:
                    prev_p, prev_old, prev_new = crossing_events[-2]
                    cur_p, cur_old, cur_new = crossing_events[-1]

                    # Determine old segment from earlier crossing to newer one
                    if prev_old < cur_old:
                        old_seg = path_list[prev_old: cur_old + 1]
                    else:
                        old_seg = path_list[cur_old: prev_old + 1][::-1]

                    # New segment includes current traversal path up to the new crossing
                    new_seg = path_list[prev_new:cur_new] + [p]

                    # Try both loop directions (forward and reverse)
                    loop_fwd = old_seg + new_seg[::-1]
                    loop_rev = new_seg + old_seg[::-1]

                    chosen_loop = None
                    for loop2 in [loop_fwd, loop_rev]:
                        if tuple(loop2) not in seen_loops and is_opposite_corners(prev_p, cur_p, loop2):
                            chosen_loop = loop2
                            break

                    if chosen_loop:
                        print(f"\n🔁 Loop {loop_id}: between-crossings created by {prev_p} & {cur_p}")
                        print(f"  Loop boundary points: {chosen_loop}")
                        enclosed2, has2, cells2 = find_enclosed_area(matrixA, chosen_loop)
                        if has2:
                            print("  Enclosed cells:")
                            for cell in sorted(cells2):
                                val = matrixA[cell[0]][cell[1]]
                                status = "visited" if cell in path_set else "new"
                                print(f"    {cell}: value={val}, {status}")
                        else:
                            print("  No enclosed 1/-1 cells.")

                        seen_loops.add(tuple(chosen_loop))
                        nas2 = handle_loop(
                            chosen_loop,
                            processed_counts,
                            entryPoint,
                            exitPoint,
                            path_list,
                            loop_id,
                            all_agents,
                            walked_turning_points,
                            cell_coverage,
                            force_agent_if_enclosed=True,
                        )

                        if nas2:
                            all_agents.update(nas2)
                            loop_id += 1

            path_list.append(p)
            path_set.add(p)
            print(f"Step to {p}: search path {path_list}")

        if direction != prev_dir:
            sec = Section(section_id, prev_turn, current, 0, [])
            sec.points = set(section_buf + [prev_turn, current])
            sections.append(sec)
            section_buf.clear()
            section_id += 1
            prev_turn, prev_dir = current, direction

        current, direction = nxt, ("col" if direction == "row" else "row")

    # Final segment to exit
    a1 = knot_manager.graph.add_point(*exitPoint, is_agent=True)
    knot_manager.register_agent(a1)

    if prev_turn != exitPoint:
        r1, c1 = prev_turn
        r2, c2 = exitPoint
        if r1 == r2:
            step = 1 if c2 > c1 else -1
            extra = [(r1, c) for c in range(c1 + step, c2 + step, step)]
        else:
            step = 1 if r2 > r1 else -1
            extra = [(r, c1) for r in range(r1 + step, r2 + step, step)]

        section_buf.extend(extra)
        sec = Section(section_id, prev_turn, exitPoint, 0, [])
        sec.points = set(section_buf + [prev_turn, exitPoint])
        sections.append(sec)


    return (
        path_list,
        {a.pos_2d() for a in knot_manager.agent_registry.values()},
        knot_manager.loop_registry,
        sections,
    )


def reduce_straight_agents(turning_points, path_list, crossing_cells):
    """
    Remove agent at turning point `b` if:
    - b is an agent
    - segments ab and bc both do not include a crossing
    """
    point_set = set(p[:2] for p in path_list)
    i = 1
    while i < len(turning_points) - 1:
        tp_prev = turning_points[i - 1]
        tp_mid = turning_points[i]
        tp_next = turning_points[i + 1]

        if not tp_mid.is_agent:
            i += 1
            continue

        a = tp_prev.point
        b = tp_mid.point
        c = tp_next.point

        # Check for crossings in segments a→b and b→c
        def get_segment_points(p1, p2):
            if p1[0] == p2[0]:
                step = 1 if p2[1] > p1[1] else -1
                return [(p1[0], col) for col in range(p1[1] + step, p2[1], step)]
            elif p1[1] == p2[1]:
                step = 1 if p2[0] > p1[0] else -1
                return [(row, p1[1]) for row in range(p1[0] + step, p2[0], step)]
            else:
                return []

        ab_cross = any(pt in crossing_cells for pt in get_segment_points(a, b))
        bc_cross = any(pt in crossing_cells for pt in get_segment_points(b, c))

        if ab_cross or bc_cross:
            i += 1
            continue

        # Remove agent at b
        tp_mid.is_agent = False
        to_remove = None
        for agent_id, point in knot_manager.agent_registry.items():
            if point.pos_2d() == b:
                to_remove = agent_id
                break
        if to_remove is not None:
            del knot_manager.agent_registry[to_remove]
            print(f"\u2796 Removed agent at {b} (turning point, no segment crossing)")

        i += 1

def segment_points_between(p1, p2):
    points = []
    if p1[0] == p2[0]:  # horizontal
        row = p1[0]
        for col in range(min(p1[1], p2[1]) + 1, max(p1[1], p2[1])):
            points.append((row, col))
    elif p1[1] == p2[1]:  # vertical
        col = p1[1]
        for row in range(min(p1[0], p2[0]) + 1, max(p1[0], p2[0])):
            points.append((row, col))
    return points


def mark_inverse_path(matrixA, entryPoint, exitPoint):
    currentPoint = exitPoint
    direction = find_starting_direction(matrixA, exitPoint[0], exitPoint[1])
    if not direction:
        return

    while currentPoint != entryPoint:
        nextPoint = search_next_turn(
            matrixA, currentPoint[0], currentPoint[1], direction
        )
        if not nextPoint:
            break

        row1, col1 = currentPoint
        row2, col2 = nextPoint

        if direction == "row":
            step = 1 if col2 > col1 else -1
            for c in range(col1 + step, col2 + step, step):
                if matrixA[row1][c] in {1, -1}:
                    continue
                matrixA[row1][c] = 3 if matrixA[row1][c] == 2 else 2
        else:
            step = 1 if row2 > row1 else -1
            for r in range(row1 + step, row2 + step, step):
                if matrixA[r][col1] in {1, -1}:
                    continue
                matrixA[r][col1] = 3 if matrixA[r][col1] == 2 else 2

        currentPoint = nextPoint
        direction = "col" if direction == "row" else "row"


def compute_agent_reduction(matrixA, entryPoint, exitPoint):
    mark_inverse_path(matrixA, entryPoint, exitPoint)

    # Perform trace and let knot_manager handle structure
    full_path, _, _, sections = trace_knot_path(matrixA, entryPoint, exitPoint)

    # Create linked list of nodes
    agent_positions = {p.pos_2d() for p in knot_manager.agent_registry.values()}
    head = None
    prev = None
    for point in full_path:
        point_type = "agent" if point in agent_positions else "path"
        node = Node(point, point_type)
        if prev:
            prev.next = node
        else:
            head = node
        prev = node

    cross_count = sum(row.count(3) for row in matrixA)

    print("\n📦 Overview of All Detected Loops:")
    for loop_id, loop_info in knot_manager.loop_registry.items():
        print(f"  🔁 Loop #{loop_id}:")
        print(f"     Path Points: {loop_info['path']}")
        print(f"     Agents: {loop_info['agents']}")

    return (
        full_path,
        head,
        cross_count,
        knot_manager.loop_registry,
        knot_manager.agent_registry,
        sections,
    )


if __name__ == "__main__":
    matrixA, entryPoint, exitPoint = read_path()

    full_path, head, cross_count, loop_map, agent_map, sections = (
        compute_agent_reduction(matrixA, entryPoint, exitPoint)
    )

    print("\n📦 Overview of All Detected Loops:")
    for loop_id, data in loop_map.items():
        print(f"\n🔁 Loop #{loop_id}:")
        print(f"  Path Points: {data['path']}")
        print(f"  Assigned Agents:")
        for agent_id, pos in data["agents"]:
            print(f"    Agent {agent_id}: {pos}")

    print("\n🧩 Final Matrix State:")
    for row in matrixA:
        print("  " + " ".join(f"{cell:2}" for cell in row))

    print("\n👥 Agent Registry:")
    for agent_id, point in agent_map.items():
        print(f"  Agent {agent_id}: {point.pos_2d()}")

    print("\n📐 Section Overview:")
    for section in sections:
        print(f"  {section}")
