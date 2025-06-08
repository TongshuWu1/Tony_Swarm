from pathNode import Node
from collections import deque
import colorsys

agent_registry = {}  # Global map to store agents with unique integer IDs
loop_registry = {}  # Global map to store loops by loop ID
next_agent_id = 1
next_loop_id = 1


# Add these classes at the top of your file
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
    global agent_registry, loop_registry, next_agent_id, next_loop_id
    agent_registry = {}
    loop_registry = {}
    next_agent_id = 1
    next_loop_id = 1


def read_path():
    rows = int(input("Enter the number of rows: "))
    cols = int(input("Enter the number of columns: "))
    matrixA = []
    print("Enter the matrix row by row (comma separated):")
    for i in range(rows):
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
    path_set = set((r, c) for r, c in looppath)
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    exterior = set()
    queue = deque()

    for r in range(rows):
        for c in [0, cols - 1]:
            if (r, c) not in path_set:
                queue.append((r, c))
        for c in range(cols):
            if (0, c) not in path_set:
                queue.append((0, c))
            if (rows - 1, c) not in path_set:
                queue.append((rows - 1, c))

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

    enclosed_area = set()
    queue = deque([start_point])
    visited = set()

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

    cells_with_values = {(r, c) for r, c in enclosed_area if matrix[r][c] in {1, -1}}
    contains_value = bool(cells_with_values)
    return enclosed_area, contains_value, cells_with_values


def handle_loop(
    matrixA,
    loop,
    processed_regions,
    entryPoint,
    exitPoint,
    path_list,
    loop_id,
    current_agents,
    loop_registry,
):
    global next_agent_id

    enclosed_area, contains_value, cells_with_values = find_enclosed_area(matrixA, loop)
    uncovered_area = enclosed_area - processed_regions
    if not uncovered_area:
        return set()
    processed_regions.update(enclosed_area)
    processed_regions.update(loop)
    agent_points = []

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
            val = matrixA[curr[0]][curr[1]]
            if is_turning_point(prev, curr, nxt) and val in {1, -1} and val != 3:
                return curr
        for offset in range(n):
            idx = (start_idx + offset) % n
            r, c = path[idx]
            val = matrixA[r][c]
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
            val = matrixA[curr[0]][curr[1]]
            if is_turning_point(prev, curr, nxt) and val in {1, -1} and val != 3:
                return curr
        return None

    # Use loop traversal order to find agents
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

    loop_registry[loop_id] = {"path": loop, "agents": []}

    for agent in new_agents:
        agent_registry[next_agent_id] = agent
        loop_registry[loop_id]["agents"].append((next_agent_id, agent))
        print(f"    Agent {next_agent_id}: {agent}")
        next_agent_id += 1

    return set(new_agents)


def trace_knot_path(matrixA, entryPoint, exitPoint):
    global next_agent_id, agent_registry, loop_registry
    reset_globals()

    currentPoint = entryPoint
    path_list = [currentPoint]
    path_set = {currentPoint}
    all_agents = set()
    processed_regions = set()
    seen_loops = set()
    loop_id = 1
    section_id = 1

    turning_points = [TurningPoint(entryPoint, is_agent=True)]
    sections = []

    agent_registry[next_agent_id] = entryPoint
    next_agent_id += 1

    direction = find_starting_direction(matrixA, entryPoint[0], entryPoint[1])
    if not direction:
        return [], set(), {}, []

    prev_turn = currentPoint
    prev_dir = direction
    section_points_buffer = []

    while currentPoint != exitPoint:
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
                pt = (row1, c)
                section_points_buffer.append(pt)
                loop = detect_loop(path_list, path_set, pt)
                if loop and tuple(loop) not in seen_loops:
                    seen_loops.add(tuple(loop))
                    new_agents = handle_loop(
                        matrixA,
                        loop,
                        processed_regions,
                        entryPoint,
                        exitPoint,
                        path_list,
                        loop_id,
                        all_agents,
                        loop_registry,
                    )
                    if new_agents:
                        all_agents.update(new_agents)
                        loop_id += 1
                path_list.append(pt)
                path_set.add(pt)
        else:  # column direction
            step = 1 if row2 > row1 else -1
            for r in range(row1 + step, row2 + step, step):
                pt = (r, col1)
                section_points_buffer.append(pt)
                loop = detect_loop(path_list, path_set, pt)
                if loop and tuple(loop) not in seen_loops:
                    seen_loops.add(tuple(loop))
                    new_agents = handle_loop(
                        matrixA,
                        loop,
                        processed_regions,
                        entryPoint,
                        exitPoint,
                        path_list,
                        loop_id,
                        all_agents,
                        loop_registry,
                    )
                    if new_agents:
                        all_agents.update(new_agents)
                        loop_id += 1
                path_list.append(pt)
                path_set.add(pt)

        if direction != prev_dir:
            is_agent = currentPoint in all_agents
            turning_points.append(TurningPoint(currentPoint, is_agent=is_agent))

            sec_points = set(section_points_buffer)
            sec_points.add(prev_turn)
            sec_points.add(currentPoint)

            sec = Section(
                section_id=section_id,
                start=prev_turn,
                end=currentPoint,
                over_under=0,
                crossings=[],
            )
            sec.points = sec_points
            sections.append(sec)

            section_points_buffer = []
            section_id += 1
            prev_turn = currentPoint
            prev_dir = direction

        currentPoint = nextPoint
        direction = "col" if direction == "row" else "row"

    # Ensure final turning point is added
    turning_points.append(TurningPoint(exitPoint, is_agent=True))
    agent_registry[next_agent_id] = exitPoint
    next_agent_id += 1

    # ✅ FINAL SECTION FIX — Only correction
    if prev_turn != exitPoint:
        row1, col1 = prev_turn
        row2, col2 = exitPoint

        if row1 == row2:
            step = 1 if col2 > col1 else -1
            for c in range(col1 + step, col2 + step, step):
                pt = (row1, c)
                section_points_buffer.append(pt)
                path_list.append(pt)
                path_set.add(pt)
        elif col1 == col2:
            step = 1 if row2 > row1 else -1
            for r in range(row1 + step, row2 + step, step):
                pt = (r, col1)
                section_points_buffer.append(pt)
                path_list.append(pt)
                path_set.add(pt)

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

    crossing_points_set = {
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
                pt in crossing_points_set
                and pt in seen_so_far  # means it's a real crossing (visited earlier)
                and pt not in used_crossings
                and pt != sec.start
                and pt != sec.end
            ):
                pending_crosses.append(pt)
                used_crossings.add(pt)
                print(f"👉 Will assign crossing {pt} to Section {sec.id + 1}")
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

    print("\n✅ Final Agent Assignment:")
    for agent_id in sorted(agent_registry):
        print(f"  Agent {agent_id}: {agent_registry[agent_id]}")

    print("\n📐 Turning Points Summary:")
    for tp in turning_points:
        print(f"  {tp}")

    print("\n🔍 Sections Summary:")
    for section in sections:
        print(f"  {section}")

    return path_list, set(agent_registry.values()), loop_registry, sections


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
    full_path, agent_points, loop_map, sections = trace_knot_path(
        matrixA, entryPoint, exitPoint
    )

    head = None
    prev = None
    for point in full_path:
        point_type = "agent" if point in agent_points else "path"
        node = Node(point, point_type)
        if prev:
            prev.next = node
        else:
            head = node
        prev = node
    cross_count = sum(row.count(3) for row in matrixA)

    # Print loop overview for GUI runs
    print("\n📦 Overview of All Detected Loops:")
    for loop_id in sorted(loop_map.keys()):
        path = loop_map[loop_id]["path"]
        agents = loop_map[loop_id]["agents"]
        print(f"  🔁 Loop #{loop_id}:")
        print(f"     Path Points: {path}")
        print(f"     Agents: {agents}")

    return full_path, head, cross_count, loop_map, agent_registry, sections


if __name__ == "__main__":
    matrixA, entryPoint, exitPoint = read_path()
    full_path, head, cross_count, loop_map = compute_agent_reduction(
        matrixA, entryPoint, exitPoint
    )

    for loop_id, data in loop_map.items():
        print(f"\nLoop #{loop_id}:")
        print(f"  Path: {data['path']}")
        print(f"  Assigned Agents: {data['agents']}")

    print("\n📦 Overview of All Detected Loops:")
    for loop_id in sorted(loop_map.keys()):
        path = loop_map[loop_id]["path"]
        agents = loop_map[loop_id]["agents"]
        print(f"  🔁 Loop #{loop_id}:")
        print(f"     Path Points: {path}")
        print(f"     Agents: {agents}")

    print("\n🧩 Final Matrix State:")
    for row in matrixA:
        print("  " + " ".join(f"{cell:2}" for cell in row))
