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
    r1, c1 = a;  r2, c2 = b;  r3, c3 = c

    # 1) both endpoints on crossings
    if matrixA[r1][c1] != 3 or matrixA[r3][c3] != 3:
        return None

    # 2) b must be a turn
    if not ((r1 == r2 == r3) or (c1 == c2 == c3)):  # i.e. not straight
        # 3) In a right‑angle pocket, the “cell in the pocket” is at (r1, c3) or (r3, c1)
        #    depending on whether you turned vertical→horizontal or vice versa.
        pocket = (r1, c3) if (r1 != r3) else (r3, c1)
        pr, pc = pocket
        if 0 <= pr < len(matrixA) and 0 <= pc < len(matrixA[0]) \
           and matrixA[pr][pc] in {1, -1}:
            # we’ve found a “pocket” – return the triple as a mini‑loop
            return [a, b, c]
    return None


def find_enclosed_area(matrix, looppath):
    rows, cols = len(matrix), len(matrix[0])
    path_set = set(looppath)
    directions = [(-1,0),(1,0),(0,-1),(0,1)]
    exterior, queue = set(), deque()
    for r in range(rows):
        for c in (0, cols-1):
            if (r,c) not in path_set:
                queue.append((r,c))
        for c in range(cols):
            for rr in (0, rows-1):
                if (rr,c) not in path_set:
                    queue.append((rr,c))
    while queue:
        r,c = queue.popleft()
        if (r,c) in exterior or (r,c) in path_set or not (0<=r<rows and 0<=c<cols):
            continue
        exterior.add((r,c))
        for dr,dc in directions:
            nr,nc = r+dr,c+dc
            queue.append((nr,nc))
    start = None
    for r in range(rows):
        for c in range(cols):
            if (r,c) not in exterior and (r,c) not in path_set:
                start = (r,c)
                break
        if start: break
    if not start:
        return set(), False, set()
    enclosed, visited, queue = set(), {start}, deque([start])
    while queue:
        r,c = queue.popleft()
        if (r,c) in enclosed or (r,c) in path_set or not (0<=r<rows and 0<=c<cols):
            continue
        enclosed.add((r,c))
        for dr,dc in directions:
            queue.append((r+dr,c+dc))
    cells = {(r,c) for (r,c) in enclosed if matrix[r][c] in {1,-1}}
    return enclosed, bool(cells), cells


def handle_loop(loop, processed_cells_map, entryPoint, exitPoint, path_list, loop_id, current_agents, walked_turning_points):
    """
    loop: list of boundary points
    processed_cells_map: dict mapping frozenset(enclosed_region) -> set of painted crossing-cells
    """
    # 1) Identify enclosed crossing-cells
    enclosed, has_cells, cells = find_enclosed_area(knot_manager.matrix, loop)
    key = frozenset(enclosed)
    max_cells = len(cells)

    # Initialize painted set for this region
    painted = processed_cells_map.setdefault(key, set())

    # Debug region status
    print(f"🚩 handle_loop loop#{loop_id}: region={key}")
    print(f"    crossing-cells in region: {cells}")
    print(f"    already painted: {painted} ({len(painted)}/{max_cells})")

    # 2) Determine new crossing-cells
    new_cells = cells - painted
    if not new_cells:
        print(f"⏩ No new crossing-cells in region {key}, skipping.")
        return set()

    # 3) Turning-point deduplication
    def turn(prev, curr, nxt):
        return (prev[0]==curr[0] and nxt[0]!=curr[0]) or (prev[1]==curr[1] and nxt[1]!=curr[1])

    tps = {loop[i] for i in range(1, len(loop)-1)
           if turn(loop[i-1], loop[i], loop[i+1])}
    reused = tps & walked_turning_points
    walked_turning_points.update(tps)
    if reused:
        print(f"⚠️ Skipping loop#{loop_id}, turning-points {reused} reused.")
        return set()

    # 4) Generate boundary candidates
    def best(idx, path):
        n = len(path)
        for off in range(n):
            p0, p1, p2 = path[(idx+off-1)%n], path[(idx+off)%n], path[(idx+off+1)%n]
            v = knot_manager.matrix[p1[0]][p1[1]]
            if turn(p0,p1,p2) and v in {1,-1}:
                return p1
        for off in range(n):
            p = path[(idx+off)%n]
            if knot_manager.matrix[p[0]][p[1]] in {1,-1}:
                return p
        return None

    def next_turn(end, path):
        n, si = len(path), path.index(end)
        for i in range(si+1, n-1):
            p0, p1, p2 = path[i-1], path[i], path[i+1]
            if p1 in loop: continue
            if turn(p0,p1,p2) and knot_manager.matrix[p1[0]][p1[1]] in {1,-1}:
                return p1
        return None

    def extreme(loop):
        sp = [p for p in loop if knot_manager.matrix[p[0]][p[1]] in {1,-1}]
        if len(sp) < 4:
            return []
        res = []
        for fn in (lambda p:p[1], lambda p:-p[1], lambda p:p[0], lambda p:-p[0]):
            for p in sorted(sp, key=fn):
                if p not in res:
                    res.append(p)
                    break
        return res

    pts = extreme(loop)
    if len(pts) < 4:
        step = max(1, len(loop)//4)
        for i in range(0, len(loop), step):
            c = best(i, loop) or next_turn(loop[-1], path_list)
            if c and c not in pts:
                pts.append(c)

    # 5) Match new_cells to boundary candidates
    new_agents = [p for p in pts if p in new_cells and p not in current_agents]

    # 6) Fallback: for each cell in new_cells, pick nearest boundary candidate
    if not new_agents:
        for cell in new_cells:
            nearest = min(pts, key=lambda p: abs(p[0]-cell[0]) + abs(p[1]-cell[1]))
            if nearest not in new_agents:
                new_agents.append(nearest)
        print(f"🔄 Fallback boundary mapping: new_cells {new_cells} -> boundary points {new_agents}")

    # 7) Assign and mark painted
    knot_manager.loop_registry[loop_id] = {"path": loop, "agents": []}
    for p in new_agents:
        aid = knot_manager.register_agent(knot_manager.graph.add_point(*p))
        knot_manager.loop_registry[loop_id]["agents"].append((aid, p))
        # mark any new_cells that map to this boundary point as painted
        for cell in list(new_cells):
            if abs(p[0]-cell[0]) + abs(p[1]-cell[1]) <= 1:
                painted.add(cell)

    print(f"✅ loop#{loop_id}: assigned agents {new_agents}, painted now {painted} ({len(painted)}/{max_cells})")
    return set(new_agents)



def trace_knot_path(matrixA, entryPoint, exitPoint):
    """
    Walk the knot from entry to exit, detecting loops, and print detailed debug info.
    Prints:
      - segment points being traced each turn,
      - each step in the evolving search path,
      - full search path at loop detection,
      - loop boundary points,
      - old_seg and new_seg for between-crossings loops,
      - enclosed cells info.
    Returns path_list, agent coords, loop registry, and sections.
    """
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

        # Compute the segment of points between current and nxt
        r1, c1 = current; r2, c2 = nxt
        if direction == 'row':
            step = 1 if c2 > c1 else -1
            pts = [(r1, c) for c in range(c1+step, c2+step, step)]
        else:
            step = 1 if r2 > r1 else -1
            pts = [(r, c1) for r in range(r1+step, r2+step, step)]

        # Debug: print segment being traced
        print(f"Tracing segment from {current} to {nxt}: {pts}")

        for p in pts:
            section_buf.append(p)

            if p in path_set:
                # Debug: full search path at loop detection
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
                    nas = handle_loop(loop1, processed_counts,
                                      entryPoint, exitPoint,
                                      path_list, loop_id,
                                      all_agents, set())
                    if nas:
                        all_agents.update(nas)
                        loop_id += 1

                # Between-crossings loop
                if len(crossing_events) >= 2:
                    prev_p, prev_old, prev_new = crossing_events[-2]
                    cur_p, cur_old, cur_new = crossing_events[-1]
                    old_seg = path_list[prev_old:cur_old+1]
                    new_seg = path_list[prev_new:cur_new] + [p]
                    print(f"  Debug old_seg: {old_seg}")
                    print(f"  Debug new_seg: {new_seg}")
                    loop2 = old_seg + new_seg[::-1]
                    print(f"\n🔁 Loop {loop_id}: between-crossings created by {prev_p} & {cur_p}")
                    print(f"  Loop boundary points: {loop2}")
                    enclosed2, has2, cells2 = find_enclosed_area(matrixA, loop2)
                    if has2:
                        print("  Enclosed cells:")
                        for cell in sorted(cells2):
                            val = matrixA[cell[0]][cell[1]]
                            status = "visited" if cell in path_set else "new"
                            print(f"    {cell}: value={val}, {status}")
                    else:
                        print("  No enclosed 1/-1 cells.")
                    if has2 and not (cells2 & path_set) and tuple(loop2) not in seen_loops:
                        seen_loops.add(tuple(loop2))
                        nas2 = handle_loop(loop2, processed_counts,
                                           entryPoint, exitPoint,
                                           path_list, loop_id,
                                           all_agents, set())
                        if nas2:
                            all_agents.update(nas2)
                            loop_id += 1

            # Mark visited and debug step
            path_list.append(p)
            path_set.add(p)
            print(f"Step to {p}: search path {path_list}")

        # Section boundary on direction change
        if direction != prev_dir:
            sec = Section(section_id, prev_turn, current, 0, [])
            sec.points = set(section_buf + [prev_turn, current])
            sections.append(sec)
            section_buf.clear()
            section_id += 1
            prev_turn, prev_dir = current, direction

        current, direction = nxt, ('col' if direction=='row' else 'row')

    # Finalize exit
    a1 = knot_manager.graph.add_point(*exitPoint, is_agent=True)
    knot_manager.register_agent(a1)
    if prev_turn != exitPoint:
        r1, c1 = prev_turn; r2, c2 = exitPoint
        if r1 == r2:
            step = 1 if c2 > c1 else -1
            extra = [(r1, c) for c in range(c1+step, c2+step, step)]
        else:
            step = 1 if r2 > r1 else -1
            extra = [(r, c1) for r in range(r1+step, r2+step, step)]
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
    i = 1
    while i < len(turning_points)-1:
        prev, mid, nxt = turning_points[i-1], turning_points[i], turning_points[i+1]
        if not mid.is_agent:
            i += 1; continue
        a, b, c = prev.point, mid.point, nxt.point
        def seg_pts(p1,p2):
            if p1[0]==p2[0]: return [(p1[0], col) for col in range(min(p1[1],p2[1])+1, max(p1[1],p2[1]))]
            if p1[1]==p2[1]: return [(row, p1[1]) for row in range(min(p1[0],p2[0])+1, max(p1[0],p2[0]))]
            return []
        if any(pt in crossing_cells for pt in seg_pts(a,b)) or any(pt in crossing_cells for pt in seg_pts(b,c)):
            i += 1; continue
        # drop agent
        mid.is_agent=False
        rem=None
        for aid, pt in list(knot_manager.agent_registry.items()):
            if pt.pos_2d()==b: rem=aid; break
        if rem is not None:
            del knot_manager.agent_registry[rem]
            print(f"➖ Removed agent at {b}—no crossing on either segment")
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
