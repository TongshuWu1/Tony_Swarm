from shapely.geometry import LineString

def adaptive_physical_layout(path_list, crossing_matrix, agent_points, cell_size=1.0):
    def to_xy(pt):
        return (pt[1] * cell_size, pt[0] * cell_size)

    segments = []
    layout_segments = []
    offset_tracker = 1

    # Split the full path into sections between agents
    def split_into_sections(path_list, agent_points):
        sections = []
        current_section = []
        for pt in path_list:
            current_section.append(pt)
            if pt[:2] in agent_points and len(current_section) > 1:
                sections.append(current_section)
                current_section = [pt]
        if len(current_section) > 1:
            sections.append(current_section)
        return sections

    def segments_intersect(seg1, seg2):
        line1 = LineString(seg1)
        line2 = LineString(seg2)
        return line1.intersects(line2) and not line1.touches(line2)

    sections = split_into_sections(path_list, agent_points)

    for idx, section in enumerate(sections):
        start = to_xy(section[0])
        end = to_xy(section[-1])
        new_seg = (start, end)

        crosses = False
        for (a, b) in segments:
            if segments_intersect(new_seg, (a, b)):
                print(f"Cross detected: new {new_seg} with existing {(a, b)}")
                crosses = True
                break

        if not crosses:
            layout_segments.append(new_seg)
            segments.append(new_seg)
        else:
            # Apply offset perpendicular to segment
            dx = end[0] - start[0]
            dy = end[1] - start[1]
            norm = (dx ** 2 + dy ** 2) ** 0.5 or 1.0
            offset = offset_tracker * 0.4 * cell_size
            offset_tracker += 1
            perp = (-dy / norm * offset, dx / norm * offset)

            displaced_start = (start[0] + perp[0], start[1] + perp[1])
            displaced_end = (end[0] + perp[0], end[1] + perp[1])

            layout_segments.append((displaced_start, displaced_end))
            segments.append((displaced_start, displaced_end))

    return layout_segments

def draw_adaptive_layout(canvas, layout_segments, scale=40.0, offset=20):
    canvas.delete("adaptive")
    for (x0, y0), (x1, y1) in layout_segments:
        canvas.create_line(
            x0 * scale + offset,
            y0 * scale + offset,
            x1 * scale + offset,
            y1 * scale + offset,
            fill="green",
            width=2,
            tags="adaptive"
        )
