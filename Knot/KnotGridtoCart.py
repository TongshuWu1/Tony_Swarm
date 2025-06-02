# KnotGridtoCart.py

from region_detection import compute_agent_reduction

# === Parameters ===
CELL_SIZE = 1.0  # meters per grid cell
ORIGIN = (0.0, 0.0)  # Cartesian origin (bottom-left of grid)


def grid_to_cartesian(grid_point, cell_size=CELL_SIZE, origin=ORIGIN):
    row, col = grid_point
    x = origin[0] + col * cell_size
    y = origin[1] - row * cell_size  # Flip Y-axis for Cartesian
    return (x, y)


def generate_cartesian_segments(matrix, entry, exit):
    grid_path, head, crossing_count = compute_agent_reduction(matrix, entry, exit)

    # Rebuild the path list from linked list
    node = head
    path_with_tags = []
    while node:
        path_with_tags.append((node.data, node.point_identifier))
        node = node.next

    # Extract agent points for segmentation
    agent_points = [pt for pt, tag in path_with_tags if tag == "agent"]
    segments = []

    i = 0
    while i < len(path_with_tags):
        pt, tag = path_with_tags[i]
        if tag == "agent":
            # Look for next agent
            for j in range(i + 1, len(path_with_tags)):
                next_pt, next_tag = path_with_tags[j]
                if next_tag == "agent":
                    intermediate = [path_with_tags[k][0] for k in range(i + 1, j)]
                    if len(intermediate) == 0:
                        segments.append(("straight", pt, next_pt))
                    else:
                        segments.append(("loop", [pt] + intermediate + [next_pt]))
                    i = j - 1
                    break
        i += 1

    # Convert to Cartesian
    cartesian_segments = []
    for seg in segments:
        if seg[0] == "straight":
            cartesian_segments.append(("straight",
                                       grid_to_cartesian(seg[1]), grid_to_cartesian(seg[2])))
        elif seg[0] == "loop":
            cartesian_segments.append(("loop", [grid_to_cartesian(p) for p in seg[1]]))

    return cartesian_segments, crossing_count


# Example usage (remove for import-based use):
if __name__ == "__main__":
    from region_detection import read_path, print_matrix

    matrix, entry, exit = read_path()
    print("\nInput Matrix:")
    print_matrix(matrix)
    segments, crosses = generate_cartesian_segments(matrix, entry, exit)
    print("\n=== Cartesian Path Representation ===")
    for seg in segments:
        print(seg)
    print(f"Total Crossings: {crosses}")