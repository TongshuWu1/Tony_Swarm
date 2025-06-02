from pathNode import Node
from collections import deque

def read_path():
    """Reads the matrix and entry/exit points from user input."""
    rows = int(input("Enter the number of rows: "))
    cols = int(input("Enter the number of columns: "))

    matrixA = []

    # Read the matrix row by row
    print("Enter the matrix row by row (comma separated):")
    for i in range(rows):
        row = list(map(int, input().split(',')))
        matrixA.append(row)

    # Validate Entry Point
    while True:
        print("Enter the path entry point (row, col):")
        entry_input = input().split(',')
        if len(entry_input) == 2:
            entryPoint = tuple(map(int, entry_input))
            if 0 <= entryPoint[0] < rows and 0 <= entryPoint[1] < cols and matrixA[entryPoint[0]][entryPoint[1]] in (-1, 1):
                print("Valid entry point:", entryPoint)
                break
            else:
                print("Invalid entry point.")
        else:
            print("Invalid input format. Use 'row,col'.")

    # Validate Exit Point
    while True:
        print("Enter the path exit point (row, col):")
        exit_input = input().split(',')
        if len(exit_input) == 2:
            exitPoint = tuple(map(int, exit_input))
            if 0 <= exitPoint[0] < rows and 0 <= exitPoint[1] < cols and matrixA[exitPoint[0]][exitPoint[1]] in (-1, 1):
                print("Valid exit point:", exitPoint)
                break
            else:
                print("Invalid exit point.")
        else:
            print("Invalid input format. Use 'row,col'.")

    return matrixA, entryPoint, exitPoint

def print_matrix(matrixA):
    """Prints the matrix in a readable format."""
    for row in matrixA:
        print(" ".join(map(str, row)))

def search_next_turn(matrixA, row, col, direction):
    """Finds the next turning point by checking both directions in a row or column."""
    rows, cols = len(matrixA), len(matrixA[0])

    if direction == "row":  # Searching left and right
        for c in range(col - 1, -1, -1):  # Left search
            if matrixA[row][c] in {1, -1}:
                return (row, c)
        for c in range(col + 1, cols):  # Right search
            if matrixA[row][c] in {1, -1}:
                return (row, c)

    elif direction == "col":  # Searching up and down
        for r in range(row - 1, -1, -1):  # Up search
            if matrixA[r][col] in {1, -1}:
                return (r, col)
        for r in range(row + 1, rows):  # Down search
            if matrixA[r][col] in {1, -1}:
                return (r, col)

    return None  # No turn found

def find_starting_direction(matrixA, row, col):
    if search_next_turn(matrixA, row, col, "row"):
        return "row"
    elif search_next_turn(matrixA, row, col, "col"):
        return "col"
    return None

def detect_loop(path_list, path_set, current_point):
    """Detects a loop in the path and returns the loop if found."""
    if current_point in path_set:
        loop_start = path_list.index(current_point)
        return path_list[loop_start:]
    return None
def handle_loop(matrixA, loop, processed_regions, entryPoint, exitPoint):
    print("\nLoop Detected!")
    print("Loop Path:", loop)

    enclosed_area, contains_value, cells_with_values = find_enclosed_area(matrixA, loop)

    # Check if the enclosed area is already processed
    uncovered_area = enclosed_area - processed_regions
    if not uncovered_area:
        print("This loop's region was already processed. Skipping.")
        return set()

    processed_regions.update(enclosed_area)
    processed_regions.update(loop)
    print("Enclosed Area:", enclosed_area)
    print("Contains 1 or -1:", contains_value)
    print("Cells containing 1 or -1:", cells_with_values)

    agent_points = set()
    print("\nAssigning agents to loop turning points with fallback:")

    # Helper: detect if the current point is a turning point
    def is_turning_point(prev, curr, nxt):
        dr1, dc1 = curr[0] - prev[0], curr[1] - prev[1]
        dr2, dc2 = nxt[0] - curr[0], nxt[1] - curr[1]
        return (dr1, dc1) != (dr2, dc2)

    # Main search: find turning point or next valid cell in loop
    def find_best_agent_point(start_idx, path):
        n = len(path)

        # First try: turning points
        for i in range(n):
            idx_prev = (start_idx + i - 1) % n
            idx_curr = (start_idx + i) % n
            idx_next = (start_idx + i + 1) % n

            prev, curr, nxt = path[idx_prev], path[idx_curr], path[idx_next]
            if is_turning_point(prev, curr, nxt) and matrixA[curr[0]][curr[1]] in {1, -1} and matrixA[curr[0]][curr[1]] != 3:
                return curr

        # Fallback: any valid cell in the loop
        for i in range(n):
            idx = (start_idx + i) % n
            r, c = path[idx]
            if matrixA[r][c] in {1, -1} and matrixA[r][c] != 3:
                return (r, c)

        return None

    # Find approximate corner index in loop path
    corner_indices = {
        "top_left": min(range(len(loop)), key=lambda i: (loop[i][0], loop[i][1])),
        "top_right": min(range(len(loop)), key=lambda i: (loop[i][0], -loop[i][1])),
        "bottom_right": max(range(len(loop)), key=lambda i: (loop[i][0], loop[i][1])),
        "bottom_left": max(range(len(loop)), key=lambda i: (loop[i][0], -loop[i][1])),
    }

    # Assign agents from corners to valid turning points or next valid
    for corner_name, idx in corner_indices.items():
        corner = find_best_agent_point(idx, loop)
        if corner:
            agent_points.add(corner)
            print(f"Assigned agent at {corner_name.replace('_', ' ')}: {corner}")

    # Always include entry and exit points if valid
    if matrixA[entryPoint[0]][entryPoint[1]] in {1, -1}:
        agent_points.add(entryPoint)
        print(f"Assigned agent at entry point: {entryPoint}")
    if matrixA[exitPoint[0]][exitPoint[1]] in {1, -1}:
        agent_points.add(exitPoint)
        print(f"Assigned agent at exit point: {exitPoint}")

    print("\nFinal Agent Points Detected:")
    for agent in agent_points:
        print(f"  Agent at: {agent}")

    return agent_points


def find_enclosed_area(matrix, looppath):
    rows, cols = len(matrix), len(matrix[0])

    path_set = set((r, c) for r, c in looppath)

    # Define directions
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    exterior = set()
    queue = deque()

    for r in range(rows):
        for c in [0, cols - 1]:  # Left & right borders
            if (r, c) not in path_set:
                queue.append((r, c))
        for c in range(cols):
            if (0, c) not in path_set:
                queue.append((0, c))
            if (rows - 1, c) not in path_set:
                queue.append((rows - 1, c))

    # Flood-fill to mark the exterior
    while queue:
        r, c = queue.popleft()
        if (r, c) in exterior or (r, c) in path_set or not (0 <= r < rows and 0 <= c < cols):
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
        return set(), False, set()  # No enclosed area found

    enclosed_area = set()
    queue = deque([start_point])
    visited = set()

    while queue:
        r, c = queue.popleft()
        if (r, c) in visited or (r, c) in path_set or not (0 <= r < rows and 0 <= c < cols):
            continue
        enclosed_area.add((r, c))
        visited.add((r, c))
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if (nr, nc) not in visited and (nr, nc) not in path_set:
                queue.append((nr, nc))

    cells_with_values = {(r, c) for r, c in enclosed_area if matrix[r][c] in {1, -1}}
    contains_value = bool(cells_with_values)

    print("\n** Debugging Information **")
    print("Exterior Marked:", exterior)
    print("Valid Start Point:", start_point)
    print("Final Enclosed Area:", enclosed_area)

    return enclosed_area, contains_value, cells_with_values

def mark_inverse_path(matrixA, entryPoint, exitPoint):
    """Marks the path from exit to entry, identifying and marking crosses."""
    currentPoint = exitPoint

    # Determine the first search direction
    direction = find_starting_direction(matrixA, exitPoint[0], exitPoint[1])
    if not direction:
        print("No valid path found from the starting point.")
        return

    while currentPoint != entryPoint:
        nextPoint = search_next_turn(matrixA, currentPoint[0], currentPoint[1], direction)

        if not nextPoint:
            print("No more path found; stopping.")
            break

        # Store all cells between turns, ensuring no duplicates
        row1, col1 = currentPoint
        row2, col2 = nextPoint

        if direction == "row":  # Moving along a row
            step = 1 if col2 > col1 else -1
            for c in range(col1 + step, col2 + step, step):  # Avoid duplicating starting point
                if matrixA[row1][c] in {1, -1}:
                    continue  # Skip cells with values 1 and -1
                if matrixA[row1][c] == 2:
                    matrixA[row1][c] = 3  # Mark crossing
                else:
                    matrixA[row1][c] = 2  # Mark path

        else:  # Moving along a column
            step = 1 if row2 > row1 else -1
            for r in range(row1 + step, row2 + step, step):  # Avoid duplicating starting point
                if matrixA[r][col1] in {1, -1}:
                    continue  # Skip cells with values 1 and -1
                if matrixA[r][col1] == 2:
                    matrixA[r][col1] = 3  # Mark crossing
                else:
                    matrixA[r][col1] = 2  # Mark path

        currentPoint = nextPoint
        direction = "col" if direction == "row" else "row"

def trace_knot_path(matrixA, entryPoint, exitPoint):
    """Traces the full rope path, ensuring all steps are recorded and detects multiple loops."""
    currentPoint = entryPoint
    path_list = [currentPoint]
    path_set = {currentPoint}
    all_agents = set()
    processed_regions = set()

    direction = find_starting_direction(matrixA, entryPoint[0], entryPoint[1])
    if not direction:
        print("No valid path found from the starting point.")
        return [], set()

    while currentPoint != exitPoint:
        nextPoint = search_next_turn(matrixA, currentPoint[0], currentPoint[1], direction)

        if not nextPoint:
            print("No more path found; stopping.")
            break

        row1, col1 = currentPoint
        row2, col2 = nextPoint

        if direction == "row":
            step = 1 if col2 > col1 else -1
            for c in range(col1 + step, col2 + step, step):
                loop = detect_loop(path_list, path_set, (row1, c))
                if loop:
                    agents = handle_loop(matrixA, loop, processed_regions, entryPoint, exitPoint)
                    all_agents.update(agents)
                path_list.append((row1, c))
                path_set.add((row1, c))
        else:
            step = 1 if row2 > row1 else -1
            for r in range(row1 + step, row2 + step, step):
                loop = detect_loop(path_list, path_set, (r, col1))
                if loop:
                    agents = handle_loop(matrixA, loop, processed_regions, entryPoint, exitPoint)
                    all_agents.update(agents)
                path_list.append((r, col1))
                path_set.add((r, col1))

        currentPoint = nextPoint
        direction = "col" if direction == "row" else "row"

    return path_list, all_agents

def compute_agent_reduction(matrixA, entryPoint, exitPoint):
    mark_inverse_path(matrixA, entryPoint, exitPoint)
    full_path, agent_points = trace_knot_path(matrixA, entryPoint, exitPoint)

    # Convert to linked list
    head = None
    prev = None
    for point in full_path:
        point_type = "agent" if point in agent_points else "path"
        node = Node(point, point_type)  # <-- ✅ pass both args
        if prev:
            prev.next = node
        else:
            head = node
        prev = node

    cross_count = sum(row.count(3) for row in matrixA)
    return full_path, head, cross_count


if __name__ == "__main__":
    matrixA, entryPoint, exitPoint = read_path()
    print("\nInitial matrix:")
    print_matrix(matrixA)
    print(f"\nEntry point: {entryPoint}")
    print(f"Exit point: {exitPoint}")

    # Mark the inverse path
    mark_inverse_path(matrixA, entryPoint, exitPoint)

    print("\nMatrix after marking inverse path and crosses:")
    print_matrix(matrixA)

    # Trace full path and detect loops
    full_path, agent_points = trace_knot_path(matrixA, entryPoint, exitPoint)

    print("\nFull Rope Path (All Cells):")
    print(full_path)

    print("\n🧠 Final Agent Points Detected:")
    for agent in agent_points:
        print(f"  Agent at: {agent}")