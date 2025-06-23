# region_detection.py

from pathNode import Node, KnotManager
from collections import deque
import colorsys

# Manager instance to replace all global state
knot_manager = KnotManager()


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
                if matrixA[start[0]][start[1]] == 1:
                    under = True
                elif matrixA[start[0]][start[1]] == -1:
                    over = True

    elif c1 == c2:  # Vertical
        for r in range(min(r1, r2) + 1, max(r1, r2)):
            if matrixA[r][c1] == 3:
                crossing_cells.add((r, c1))
                if matrixA[start[0]][start[1]] == 1:
                    under = True
                elif matrixA[start[0]][start[1]] == -1:
                    over = True

    if over and not under:
        return 1, crossing_cells
    elif under and not over:
        return -1, crossing_cells
    elif over and under:
        print(f"⚠️ Conflicting crossing behavior: {start}->{end}")
        return 0, crossing_cells
    else:
        return 0, crossing_cells  # No crossing


def search_next_turn(matrixA, row, col, direction):
    rows, cols = len(matrixA), len(matrixA[0])
    if direction == "row":
        for c in range(col - 1, -1, -1):
            if matrixA[row][c] in {1, -1}:
                return (row, c)
        for c in range(col + 1, cols):
            if matrixA[row][c] in {1, -1}:
                return (row, c)
    elif direction == "col":
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
    elif search_next_turn(matrixA, row, col, "col"):
        return "col"
    return None


def detect_loop(path_list, path_set, current_point):
    if current_point in path_set:
        loop_start = path_list.index(current_point)
        return path_list[loop_start:]
    return None


def find_enclosed_area(matrix, looppath):
    rows, cols = len(matrix), len(matrix[0])
    path_set = set(looppath)
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    exterior = set()
    queue = deque()

    # Add border points not in path
    for r in range(rows):
        for c in [0, cols - 1]:
            if (r, c) not in path_set:
                queue.append((r, c))
        for c in range(cols):
            if (0, c) not in path_set:
                queue.append((0, c))
            if (rows - 1, c) not in path_set:
                queue.append((rows - 1, c))

    # Flood fill exterior
    while queue:
        r, c = queue.popleft()
        if (
            (r, c) in exterior
            or (r, c) in path_set
            or not (0 <= r < rows and 0 <= c < cols)
        ):
            continue
        exterior.add((r, c))
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if (nr, nc) not in exterior and (nr, nc) not in path_set:
                queue.append((nr, nc))

    # Find any point inside
    start_point = None
    for r in range(rows):
        for c in range(cols):
            if (r, c) not in exterior and (r, c) not in path_set:
                start_point = (r, c)
                break
        if start_point:
            break

    if not start_point:
        return set(), False, set()

    # Flood fill interior
    enclosed_area = set()
    visited = set()
    queue = deque([start_point])

    while queue:
        r, c = queue.popleft()
        if (
            (r, c) in visited
            or (r, c) in path_set
            or not (0 <= r < rows and 0 <= c < cols)
        ):
            continue
        enclosed_area.add((r, c))
        visited.add((r, c))
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if (nr, nc) not in visited and (nr, nc) not in path_set:
                queue.append((nr, nc))

    # Check if there's path value inside
    cells_with_values = {(r, c) for (r, c) in enclosed_area if matrix[r][c] in {1, -1}}
    contains_value = bool(cells_with_values)

    return enclosed_area, contains_value, cells_with_values


def handle_loop(
    loop,
    processed_regions,
    entryPoint,
    exitPoint,
    path_list,
    loop_id,
    current_agents,
):
    enclosed_area, contains_value, cells_with_values = find_enclosed_area(
        knot_manager.matrix, loop
    )
    uncovered_area = enclosed_area - processed_regions
    if not uncovered_area:
        return set()

    processed_regions.update(enclosed_area)
    processed_regions.update(loop)

    def is_turning_point(prev, curr, nxt):
        dr1, dc1 = curr[0] - prev[0], curr[1] - prev[1]
        dr2, dc2 = nxt[0] - curr[0], nxt[1] - curr[1]
        return (dr1, dc1) != (dr2, dc2)

    def find_best_agent_point(start_idx, path):
        n = len(path)
        for offset in range(n):
            idx_prev = (start_idx + offset - 1) % n
            idx_curr = (start_idx + offset) % n
            idx_next = (start_idx + offset + 1) % n
            prev, curr, nxt = path[idx_prev], path[idx_curr], path[idx_next]
            val = knot_manager.matrix[curr[0]][curr[1]]
            if is_turning_point(prev, curr, nxt) and val in {1, -1} and val != 3:
                return curr
        for offset in range(n):
            idx = (start_idx + offset) % n
            r, c = path[idx]
            val = knot_manager.matrix[r][c]
            if val in {1, -1} and val != 3:
                return (r, c)
        return None

    def find_next_valid_turn_after_loop(loop_end, path_list):
        n = len(path_list)
        start_idx = path_list.index(loop_end)
        for i in range(1, n - start_idx - 1):
            idx = start_idx + i
            prev, curr, nxt = path_list[idx - 1], path_list[idx], path_list[idx + 1]
            if curr in loop:
                continue
            val = knot_manager.matrix[curr[0]][curr[1]]
            if is_turning_point(prev, curr, nxt) and val in {1, -1} and val != 3:
                return curr
        return None

    def select_priority_agents(loop):
        strand_points = [pt for pt in loop if knot_manager.matrix[pt[0]][pt[1]] in {1, -1}]
        if len(strand_points) < 4:
            return []

        extremes = []
        for key_func in [
            lambda p: p[1],  # min x
            lambda p: -p[1], # max x
            lambda p: p[0],  # min y
            lambda p: -p[0], # max y
        ]:
            sorted_pts = sorted(strand_points, key=key_func)
            for pt in sorted_pts:
                if pt not in extremes:
                    extremes.append(pt)
                    break

        return extremes

    # First attempt: use geometric extreme corners
    agent_points = select_priority_agents(loop)

    # Fallback if not enough points
    if len(agent_points) < 4:
        MAX_AGENTS_PER_LOOP = 4
        step_size = max(1, len(loop) // MAX_AGENTS_PER_LOOP)
        for idx in range(0, len(loop), step_size):
            corner = find_best_agent_point(idx, loop)
            if not corner:
                loop_exit = loop[-1]
                corner = find_next_valid_turn_after_loop(loop_exit, path_list)
            if corner and corner not in agent_points:
                agent_points.append(corner)

    new_agents = [
        pt
        for pt in agent_points
        if pt not in current_agents and pt != entryPoint and pt != exitPoint
    ]
    if not new_agents:
        return set()

    print(f"\n🔁 Loop #{loop_id} Detected:")
    print("  Path:", loop)
    print("  New Assigned Agents:")

    knot_manager.loop_registry[loop_id] = {"path": loop, "agents": []}

    for agent in new_agents:
        agent_id = knot_manager.register_agent(knot_manager.graph.add_point(*agent))
        knot_manager.loop_registry[loop_id]["agents"].append((agent_id, agent))
        print(f"    Agent {agent_id}: {agent}")

    return set(new_agents)

# Full corrected `trace_knot_path` function with accurate turning point detection

def trace_knot_path(matrixA, entryPoint, exitPoint):
    knot_manager.reset()
    knot_manager.set_matrix(matrixA, entryPoint, exitPoint)

    currentPoint = entryPoint
    path_list = [currentPoint]
    path_set = {currentPoint}
    all_agents = set()
    processed_regions = set()
    seen_loops = set()
    loop_id = 1
    section_id = 1

    sections = []

    agent_entry = knot_manager.graph.add_point(*entryPoint, is_agent=True)
    agent_id = knot_manager.register_agent(agent_entry)

    direction = find_starting_direction(matrixA, *entryPoint)
    if not direction:
        return [], set(), {}, []

    prev_turn = currentPoint
    prev_dir = direction
    section_points_buffer = []

    while currentPoint != exitPoint:
        nextPoint = search_next_turn(matrixA, *currentPoint, direction)
        if not nextPoint:
            break

        row1, col1 = currentPoint
        row2, col2 = nextPoint

        intermediate_points = []
        if direction == "row":
            step = 1 if col2 > col1 else -1
            intermediate_points = [(row1, c) for c in range(col1 + step, col2 + step, step)]
        else:
            step = 1 if row2 > row1 else -1
            intermediate_points = [(r, col1) for r in range(row1 + step, row2 + step, step)]

        for pt in intermediate_points:
            section_points_buffer.append(pt)
            loop = detect_loop(path_list, path_set, pt)
            if loop and tuple(loop) not in seen_loops:
                seen_loops.add(tuple(loop))
                new_agents = handle_loop(loop, processed_regions, entryPoint, exitPoint, path_list, loop_id, all_agents)
                if new_agents:
                    all_agents.update(new_agents)
                    loop_id += 1
            path_list.append(pt)
            path_set.add(pt)

        if direction != prev_dir:
            sec = Section(
                section_id=section_id,
                start=prev_turn,
                end=currentPoint,
                over_under=0,
                crossings=[],
            )
            sec.points = set(section_points_buffer + [prev_turn, currentPoint])
            sections.append(sec)

            section_points_buffer = []
            section_id += 1
            prev_turn = currentPoint
            prev_dir = direction

        currentPoint = nextPoint
        direction = "col" if direction == "row" else "row"

    # Final agent at exit
    agent_exit = knot_manager.graph.add_point(*exitPoint, is_agent=True)
    knot_manager.register_agent(agent_exit)

    if prev_turn != exitPoint:
        row1, col1 = prev_turn
        row2, col2 = exitPoint
        if row1 == row2:
            step = 1 if col2 > col1 else -1
            section_points_buffer += [(row1, c) for c in range(col1 + step, col2 + step, step)]
        elif col1 == col2:
            step = 1 if row2 > row1 else -1
            section_points_buffer += [(r, col1) for r in range(row1 + step, row2 + step, step)]

        sec = Section(
            section_id=section_id,
            start=prev_turn,
            end=exitPoint,
            over_under=0,
            crossings=[],
        )
        sec.points = set(section_points_buffer + [prev_turn, exitPoint])
        sections.append(sec)
        section_id += 1

    # Mark crossings
    crossing_cells = {
        (r, c)
        for r in range(len(matrixA))
        for c in range(len(matrixA[0]))
        if matrixA[r][c] == 3
    }
    used_crossings = set()
    seen_so_far = set()
    pending_crosses = []

    for sec in sections:
        crosses = []
        crosses.extend(pending_crosses)
        pending_crosses = []
        for pt in sec.points:
            if (
                pt in crossing_cells
                and pt in seen_so_far
                and pt not in used_crossings
                and pt != sec.start
                and pt != sec.end
            ):
                pending_crosses.append(pt)
                used_crossings.add(pt)
        sec.crossings = sorted(crosses)
        seen_so_far.update(sec.points)

        val_start = matrixA[sec.start[0]][sec.start[1]]
        val_end = matrixA[sec.end[0]][sec.end[1]]
        if val_start == 1 and val_end == -1:
            sec.over_under = -1
        elif val_start == -1 and val_end == 1:
            sec.over_under = 1
        else:
            sec.over_under = 0

    # Build turning points from path and agent locations
    turning_points = []
    agent_coords = {agent.pos_2d() for agent in knot_manager.agent_registry.values()}
    for i, pt in enumerate(path_list):
        is_turn = False
        if 0 < i < len(path_list) - 1:
            prev = path_list[i - 1]
            nxt = path_list[i + 1]
            if (prev[0] == pt[0] and nxt[0] != pt[0]) or (prev[1] == pt[1] and nxt[1] != pt[1]):
                is_turn = True

        if pt in agent_coords:
            turning_points.append(TurningPoint(pt, is_agent=True))
        elif is_turn:
            turning_points.append(TurningPoint(pt, is_agent=False))

    print("\n✅ Final Agent Assignment:")
    for agent_id, point in knot_manager.agent_registry.items():
        print(f"  Agent {agent_id}: ({point.row},{point.col})")

    print("\n📐 Turning Points Summary:")
    for tp in turning_points:
        row, col = tp.point
        label = "Agent" if tp.is_agent else "Turn"
        print(f"  {label}@({row}, {col})")

    print("\n🔍 Sections Summary:")
    for section in sections:
        print(f"  {section}")

    return (
        path_list,
        {agent.pos_2d() for agent in knot_manager.agent_registry.values()},
        knot_manager.loop_registry,
        sections,
    )

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
