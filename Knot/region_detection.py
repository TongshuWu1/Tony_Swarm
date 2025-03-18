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
def handle_loop(matrixA, loop):
    print("\nLoop Detected! ")
    print("Loop Path:", loop)
    enclosed_area, contains_value, cells_with_values = find_enclosed_area(matrixA, loop)
    print("Enclosed Area:", enclosed_area)
    print("Contains 1 or -1:", contains_value)
    print("Cells containing 1 or -1:", cells_with_values)


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
    # print("Path Set:", path_set)
    print("Exterior Marked:", exterior)
    print("Valid Start Point:", start_point)
    print("Final Enclosed Area:", enclosed_area)

    return enclosed_area, contains_value, cells_with_values



def trace_knot_path(matrixA, entryPoint, exitPoint):
    """Traces the full rope path, ensuring all steps are recorded and detects multiple loops."""
    currentPoint = entryPoint
    path_list = []  # Store full path
    path_set = set()  # Store visited points for quick loop detection
    checked_cells = set()  # Store cells that have been checked for loops

    # Determine the first search direction
    direction = find_starting_direction(matrixA, entryPoint[0], entryPoint[1])
    if not direction:
        print("No valid path found from the starting point.")
        return []

    while currentPoint != exitPoint:
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
                loop = detect_loop(path_list, path_set, (row1, c))
                if loop:
                    handle_loop(matrixA, loop)
                path_list.append((row1, c))
                path_set.add((row1, c))

        else:  # Moving along a column
            step = 1 if row2 > row1 else -1
            for r in range(row1 + step, row2 + step, step):  # Avoid duplicating starting point
                loop = detect_loop(path_list, path_set, (r, col1))
                if loop:
                    handle_loop(matrixA, loop)
                path_list.append((r, col1))
                path_set.add((r, col1))

        currentPoint = nextPoint
        direction = "col" if direction == "row" else "row"  # Switch between row/col search

    return path_list




if __name__ == "__main__":
    matrixA, entryPoint, exitPoint = read_path()
    print("\nInitial matrix:")
    print_matrix(matrixA)
    print(f"\nEntry point: {entryPoint}")
    print(f"Exit point: {exitPoint}")

    # Compute the full rope path
    full_path = trace_knot_path(matrixA, entryPoint, exitPoint)

    print("\nFull Rope Path (All Cells):")
    print(full_path)  # Print the list of all path cells
