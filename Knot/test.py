from pathNode import Node


def read_path():
    global entryPoint, exitPoint
    rows = int(input("Enter the number of rows: "))
    cols = int(input("Enter the number of columns: "))

    matrixA = []

    # Read the matrix row by row
    print("Enter the matrix row by row (comma separated):")
    for i in range(rows):
        row = list(map(int, input().split(',')))
        matrixA.append(row)

    entry_valid = False
    while not entry_valid:
        print("Enter the path entry point (row, col):")
        entry_input = input().split(',')
        if len(entry_input) == 2:
            entryPoint = tuple(map(int, entry_input))
            if (entryPoint[0] < 0 or entryPoint[0] >= rows or
                    entryPoint[1] < 0 or entryPoint[1] >= cols or
                    matrixA[entryPoint[0]][entryPoint[1]] not in (-1, 1)):
                print("Invalid entry point:", entryPoint)
            else:
                print("Valid entry point:", entryPoint)
                entry_valid = True
        else:
            print("Invalid input. Please enter the path entry point as 'row,col'.")

    exit_valid = False
    while not exit_valid:
        print("Enter the path exit point (row, col):")
        exit_input = input().split(',')
        if len(exit_input) == 2:
            exitPoint = tuple(map(int, exit_input))
            if (exitPoint[0] < 0 or exitPoint[0] >= rows or
                    exitPoint[1] < 0 or exitPoint[1] >= cols or
                    matrixA[exitPoint[0]][exitPoint[1]] not in (-1, 1)):
                print("Invalid exit point:", exitPoint)
            else:
                print("Valid exit point:", exitPoint)
                exit_valid = True
        else:
            print("Invalid input. Please enter the path exit point as 'row,col'.")

    return matrixA, entryPoint, exitPoint


def print_matrix(matrixA):
    for row in matrixA:
        print(" ".join(map(str, row)))


def search_path(matrixA, currentPoint, pathflag):
    rows, cols = len(matrixA), len(matrixA[0])
    row0, col0 = currentPoint
    nextPoint = None
    crossing_cell = None

    def process_path(cells):
        nonlocal crossing_cell
        for (r, c) in cells:
            if matrixA[r][c] == 2:  # Crossing detected
                crossing_cell = (r, c)
                print(f"Crossing detected at {(r, c)} (marked as 3)")
                break  # Stop after marking the first crossing
            elif matrixA[r][c] == 0:  # Mark new path
                matrixA[r][c] = 2

    if pathflag in ('i', 'r'):
        for col in range(col0 + 1, cols):
            if matrixA[row0][col] in (1, -1):  # Stop at barriers
                nextPoint = (row0, col)
                process_path([(row0, c) for c in range(col0, col)])
                return nextPoint, 'c', crossing_cell

    if pathflag in ('i', 'c'):
        for row in range(row0 + 1, rows):
            if matrixA[row][col0] in (1, -1):  # Stop at barriers
                nextPoint = (row, col0)
                process_path([(r, col0) for r in range(row0, row)])
                return nextPoint, 'r', crossing_cell

    return None, 'i', crossing_cell


def compute_crossings(matrix, entry, exit_):
    print("\nRunning Path Detection...")

    currentPoint = exit_
    pathflag = "i"

    while currentPoint != entry:
        nextPoint, pathflag, crossing_cell = search_path(matrix, currentPoint, pathflag)

        if not nextPoint:
            print("No more path found; stopping.")
            break

        if crossing_cell:
            r, c = crossing_cell
            matrix[r][c] = 3  # Mark the crossing point as 3

        currentPoint = nextPoint


if __name__ == "__main__":
    matrixA, entryPoint, exitPoint = read_path()

    print("\nInitial matrix:")
    print_matrix(matrixA)

    compute_crossings(matrixA, entryPoint, exitPoint)

    print("\nFinal matrix with paths marked as 2 and crossings as 3:")
    print_matrix(matrixA)
