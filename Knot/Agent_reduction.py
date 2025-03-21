from pathNode import Node

def read_path():
    global entryPoint, exitPoint
    rows = int(input("Enter the number of rows: "))
    cols = int(input("Enter the number of columns: "))

    matrixA = []

    # Read the matrix row by row
    print("Enter the matrix row by row (comma separated):")
    for i in range(rows):
        row = list(map(int, input().replace(',', ',').split(',')))
        matrixA.append(row)

    entry_valid = False
    while not entry_valid:
        print("Enter the path entry point (row, col):")
        entry_input = input().replace(',', ',').split(',')
        if len(entry_input) == 2:
            entryPoint = tuple(map(int, entry_input))
            if (entryPoint[0] < 0 or entryPoint[0] >= rows or
                    entryPoint[1] < 0 or entryPoint[1] >= cols or
                    matrixA[entryPoint[0]][entryPoint[1]] not in (-1, 1)):
                print("Invalid entry point:", entryPoint)
                print("value: ", matrixA[entryPoint[0]][entryPoint[1]])
            else:
                print("Valid entry point:", entryPoint)
                entry_valid = True
        else:
            print("Invalid input. Please enter the path entry point as 'row,col'.")

    exit_valid = False
    while not exit_valid:
        print("Enter the path exit point (row, col):")
        exit_input = input().replace(',', ',').split(',')
        if len(exit_input) == 2:
            exitPoint = tuple(map(int, exit_input))
            if (exitPoint[0] < 0 or exitPoint[0] >= rows or
                    exitPoint[1] < 0 or exitPoint[1] >= cols or
                    matrixA[exitPoint[0]][exitPoint[1]] not in (-1, 1)):
                print("Invalid exit point:", exitPoint)
                print("value: ", matrixA[exitPoint[0]][exitPoint[1]])
            else:
                print("Valid exit point:", exitPoint)
                exit_valid = True
        else:
            print("Invalid input. Please enter the path exit point as 'row,col'.")

    return matrixA, entryPoint, exitPoint

def print_matrix(matrixA):
    for row in matrixA:
        print(" ".join(map(str, row)))

def search_path(matrixA, currentPoint, pathflag, pathindex, previousCrossIndex, crossNumber):
    print(f"Searching from {currentPoint} in direction {pathflag}")
    rows = len(matrixA)
    cols = len(matrixA[0])
    row0, col0 = currentPoint

    add_agent = 0
    nextPoint = None
    path_cells = []

    def walk_cells(cells, previousCrossIndex, crossNumber):
        nonlocal add_agent
        print("previousCrossIndex:", previousCrossIndex)
        for (r, c) in cells:
            if matrixA[r][c] == 0:
                path_cells.append((r, c))
                matrixA[r][c] = pathindex  # Mark path cells with the current pathindex
            elif matrixA[r][c] == previousCrossIndex:
                print(f"Zigzag detected at {(r, c)}")
                crossNumber += 1
                add_agent = 0  # Set add_agent to 0 when a zigzag is detected
            elif (matrixA[r][c] not in (0, 1, -1) and matrixA[r][c] != previousCrossIndex):
                add_agent = 1
                crossNumber += 1
                previousCrossIndex = matrixA[r][c]
                print(f"Crossing {previousCrossIndex} at {(r, c)}. Updating previousCrossIndex to {previousCrossIndex}. crossing: {crossNumber}")
                matrixA[r][c] = 33  # Mark crossing point as 33

        return add_agent, previousCrossIndex, crossNumber

    if pathflag == 'r':
        for col in range(cols):
            if col != col0 and matrixA[row0][col] in (1, -1):
                nextPoint = (row0, col)
                step = 1 if col > col0 else -1
                cells = [(row0, c) for c in range(col0, col + step, step)]
                add_agent, previousCrossIndex, crossNumber = walk_cells(cells, previousCrossIndex, crossNumber)
                return nextPoint, 'c', add_agent, path_cells, previousCrossIndex, crossNumber
        return None, 'r', add_agent, path_cells, previousCrossIndex, crossNumber

    elif pathflag == 'c':
        for row in range(rows):
            if row != row0 and matrixA[row][col0] in (1, -1):
                nextPoint = (row, col0)
                step = 1 if row > row0 else -1
                cells = [(r, col0) for r in range(row0, row + step, step)]
                add_agent, previousCrossIndex, crossNumber = walk_cells(cells, previousCrossIndex, crossNumber)
                return nextPoint, 'r', add_agent, path_cells, previousCrossIndex, crossNumber
        return None, 'c', add_agent, path_cells, previousCrossIndex, crossNumber

    else:
        for col in range(cols):
            if col != col0 and matrixA[row0][col] in (1, -1):
                nextPoint = (row0, col)
                step = 1 if col > col0 else -1
                cells = [(row0, c) for c in range(col0, col + step, step)]
                add_agent, previousCrossIndex, crossNumber = walk_cells(cells, previousCrossIndex, crossNumber)
                return nextPoint, 'c', add_agent, path_cells, previousCrossIndex, crossNumber

        for row in range(rows):
            if row != row0 and matrixA[row][col0] in (1, -1):
                nextPoint = (row, col0)
                step = 1 if row > row0 else -1
                cells = [(r, col0) for r in range(row0, row + step, step)]
                add_agent, previousCrossIndex, crossNumber = walk_cells(cells, previousCrossIndex, crossNumber)
                return nextPoint, 'r', add_agent, path_cells, previousCrossIndex, crossNumber

        return None, 'i', add_agent, path_cells, previousCrossIndex, crossNumber

def compute_agent_reduction(matrix, entry, exit_):
    print("\nRunning Agent Reduction Algorithm...")

    pathflag = "i"
    currentPoint = exit_
    head = Node(currentPoint, "agent")
    currentNode = head

    path_list = []
    path_list.append(exit_)

    pathindex = 2
    previousCrossIndex = -2

    crossNumber = 0

    while currentPoint != entry:
        print(f"\nCurrentPoint: {currentPoint}, pathflag: {pathflag}, pathindex: {pathindex}")
        nextPoint, pathflag, add_agent, path_cells, previousCrossIndex, crossNumber = search_path(
            matrix, currentPoint, pathflag,
            pathindex=pathindex,
            previousCrossIndex=previousCrossIndex,
            crossNumber=crossNumber
        )

        if not nextPoint:
            print("No more path found; stopping.")
            break

        if add_agent == 1:
            print(f"Crossing detected -> incrementing pathindex from {pathindex} to {pathindex + 1}")
            currentNode.point_identifier = "agent"
            pathindex += 1

        for (r, c) in path_cells:
            path_list.append((r, c))  # Record every single point

        if nextPoint == entry or add_agent == 1:
            newNode = Node(nextPoint, "agent")
        else:
            newNode = Node(nextPoint, "path")

        currentNode.next = newNode
        currentNode = newNode
        currentPoint = nextPoint

        path_list.append(nextPoint)

        if add_agent == 1:
            pathindex += 1
        print_matrix(matrix)

    return path_list, head, crossNumber

if __name__ == "__main__":
    matrixA, entryPoint, exitPoint = read_path()
    print("\nInitial matrix:")
    print_matrix(matrixA)
    print(f"\nEntry point: {entryPoint} (Agent)")
    print(f"Exit point: {exitPoint} (Agent)")

    # Compute the agent reduction path
    path, head, crossNumber = compute_agent_reduction(matrixA, entryPoint, exitPoint)

    print("\nFinal matrix:")
    print_matrix(matrixA)

    print("\nLinked path:")
    node_ptr = head
    while node_ptr:
        node_ptr.print_node()
        node_ptr = node_ptr.next

    # For debugging, show the collected path list
    print("\nPath List (from exit -> entry):")
    print(path)

    print("\nCrossings:", crossNumber)
