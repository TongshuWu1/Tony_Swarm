from shapely.geometry import LineString, Point

def check_preserve_crossings_and_update_gaps(points, segments, moved_idx, tolerance=1.5):
    """
    Checks if the segment crossings are preserved after moving a point.
    If preserved, updates the gap positions.
    Returns (True, message) or (False, error_description).
    """

    def get_segment_lines(ps):
        return [
            (seg, LineString([ps[seg.p1].pos, ps[seg.p2].pos]))
            for seg in segments
        ]

    old_segments = [
        LineString([points[s.p1].pos, points[s.p2].pos])
        for s in segments
    ]
    old_crossings = {}
    for i in range(len(old_segments)):
        for j in range(i + 1, len(old_segments)):
            if old_segments[i].crosses(old_segments[j]):
                pt = old_segments[i].intersection(old_segments[j])
                if pt.geom_type == "Point":
                    key = frozenset([i, j])
                    old_crossings[key] = pt

    new_lines = get_segment_lines(points)
    new_crossings = {}

    for i in range(len(new_lines)):
        seg_i, line_i = new_lines[i]
        for j in range(i + 1, len(new_lines)):
            seg_j, line_j = new_lines[j]
            if line_i.crosses(line_j):
                pt = line_i.intersection(line_j)
                if pt.geom_type == "Point":
                    key = frozenset([seg_i.id, seg_j.id])
                    new_crossings[key] = pt

    for key, old_pt in old_crossings.items():
        if key not in new_crossings:
            return False, f"Violated crossing between segments {key}."

    # If we reach here, all previous crossings are preserved
    # Update gap positions
    for seg in segments:
        seg.gap_at.clear()

    for key, pt in new_crossings.items():
        id1, id2 = list(key)
        seg1 = next(s for s in segments if s.id == id1)
        seg2 = next(s for s in segments if s.id == id2)
        if seg1.is_overpass:
            seg2.gap_at.append(pt)
        else:
            seg1.gap_at.append(pt)

    return True, "Gaps updated."
