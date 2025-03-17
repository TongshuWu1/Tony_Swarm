import numpy as np
from collections import deque

def find_enclosed_area(matrix, path):
    rows, cols = len(matrix), len(matrix[0])

    # Convert path list to a set for quick lookup (0-based indexing)
    path_set = set((r, c) for r, c in path)

    # Define directions (up, down, left, right)
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    ### **Step 1: Mark all external areas using flood-fill**
    exterior = set()
    queue = deque()

    # Add all boundary cells to the queue (except path points)
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

    ### **Step 2: Find a truly inside point**
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

    ### **Step 3: Flood-fill to find the enclosed area**
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

    ### **Step 4: Find cells containing 1 or -1 inside the enclosed area**
    cells_with_values = {(r, c) for r, c in enclosed_area if matrix[r][c] in {1, -1}}
    contains_value = bool(cells_with_values)

    ### **Debugging output**
    print("\n** Debugging Information **")
    print("Path Set:", path_set)
    print("Exterior Marked:", exterior)
    print("Valid Start Point:", start_point)
    print("Final Enclosed Area:", enclosed_area)

    return enclosed_area, contains_value, cells_with_values


# Example matrix
matrix = np.array([
    [0, 0, 0, 0, 0, 1, 0],
    [0, -1, 0, 0, 0, 0, 1],
    [1, 0, -1, 0, 0, 0, 0],
    [0, 1, 0, -1, 0, 0, 0],
    [0, 0, 1, 0, -1, 0, 0],
    [0, 0, 0, 1, 0, -1, 0],
    [0, 0, 0, 0, 1, 0, -1]
])

# Adjusted path (0-based indexing)
path = [
    (2, 1), (2, 2), (3, 2), (4, 2), (4, 3), (4, 4),
    (5, 4), (6, 4), (6, 5), (6, 6), (5, 6), (4, 6),
    (3, 6), (2, 6), (1, 6), (1, 5), (1, 4), (1, 3),
    (1, 2), (1, 1), (2, 1)
]

# Run the function
area, has_value, cells_with_values = find_enclosed_area(matrix, path)

# Output results
print("\nEnclosed Area:", area)
print("Contains 1 or -1:", has_value)
print("Cells containing 1 or -1:", cells_with_values)
