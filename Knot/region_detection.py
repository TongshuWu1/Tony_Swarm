from pathNode import Node

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
    """Determines the initial movement direction by checking both row and column."""
    if search_next_turn(matrixA, row, col, "row"):
        return "row"
    elif search_next_turn(matrixA, row, col, "col"):
        return "col"
    return None

def trace_knot_path(matrixA, entryPoint, exitPoint):
    """Traces the full rope path, ensuring all steps are recorded and detects multiple loops."""
    currentPoint = entryPoint
    path_list = []  # Store full path
    path_set = set()  # Store visited points for quick loop detection

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

        # **Store all cells between turns, ensuring no duplicates**
        row1, col1 = currentPoint
        row2, col2 = nextPoint

        if direction == "row":  # Moving along a row
            step = 1 if col2 > col1 else -1
            for c in range(col1 + step, col2 + step, step):  # Avoid duplicating starting point
                if (row1, c) in path_set:
                    loop_start = path_list.index((row1, c))
                    loop = path_list[loop_start:]
                    print("\nLoop Detected! ")
                    print("Loop Path:", loop)
                path_list.append((row1, c))
                path_set.add((row1, c))

        else:  # Moving along a column
            step = 1 if row2 > row1 else -1
            for r in range(row1 + step, row2 + step, step):  # Avoid duplicating starting point
                if (r, col1) in path_set:
                    loop_start = path_list.index((r, col1))
                    loop = path_list[loop_start:]
                    print("\n Loop Detected! ")
                    print("Loop Path:", loop)
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
