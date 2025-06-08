import tkinter as tk
from shapely.geometry import LineString, Point


class KnotPoint:
    def __init__(self, point_id, x, y, is_agent):
        self.id = point_id
        self.pos = (x, y)
        self.is_agent = is_agent


class KnotSegment:
    def __init__(self, seg_id, p1_id, p2_id, is_overpass):
        self.id = seg_id
        self.p1 = p1_id
        self.p2 = p2_id
        self.is_overpass = is_overpass
        self.gap_at = []


def check_preserve_crossings_and_update_gaps(
    points, segments, moved_idx, tolerance=1.5
):
    def seg_line(seg):
        return LineString([points[seg.p1].pos, points[seg.p2].pos])

    moved_segments = [s for s in segments if s.p1 == moved_idx or s.p2 == moved_idx]

    original_lines = {
        s.id: LineString([points[s.p1].pos, points[s.p2].pos]) for s in segments
    }
    original_crossings = {}
    for s in moved_segments:
        original_crossings[s.id] = set()
        for other in segments:
            if s.id != other.id and original_lines[s.id].crosses(
                original_lines[other.id]
            ):
                original_crossings[s.id].add(other.id)

    new_crossings = {}
    for s in moved_segments:
        new_crossings[s.id] = set()
        new_line = seg_line(s)
        for other in segments:
            if s.id != other.id:
                other_line = seg_line(other)
                if new_line.crosses(other_line):
                    new_crossings[s.id].add(other.id)

    for s in moved_segments:
        before = original_crossings[s.id]
        after = new_crossings[s.id]
        if before != after:
            added = after - before
            removed = before - after
            return (
                False,
                f"❌ Segment {s.id} crossing set mismatch. Added: {added}, Removed: {removed}",
            )

    for seg in segments:
        seg.gap_at.clear()

    for i in range(len(segments)):
        seg1 = segments[i]
        line1 = seg_line(seg1)
        for j in range(i + 1, len(segments)):
            seg2 = segments[j]
            line2 = seg_line(seg2)
            if line1.crosses(line2):
                pt = line1.intersection(line2)
                if pt.geom_type == "Point":
                    if seg1.is_overpass:
                        seg2.gap_at.append(pt)
                    else:
                        seg1.gap_at.append(pt)

    return True, "✅ Crossing sets match. Gaps updated."


class ShapelyGUI:
    def __init__(self, parent):
        self.parent = parent
        self.canvas = tk.Canvas(parent, width=600, height=600, bg="white")
        self.canvas.pack()

        self.points = []
        self.segments = []
        self.next_point_id = 0
        self.next_segment_id = 0

        self.dragging_point = None
        self.original_points = []

        self.show_waypoints = True

        self.canvas.bind("<ButtonPress-1>", self.on_drag_start)
        self.canvas.bind("<B1-Motion>", self.on_drag_motion)
        self.canvas.bind("<ButtonRelease-1>", self.on_drag_end)
        self.canvas.focus_set()

        self.toggle_waypoint_btn = tk.Button(
            parent, text="Hide Waypoints", command=self.toggle_waypoints
        )
        self.toggle_waypoint_btn.pack(pady=2)

        self.clear_button = tk.Button(parent, text="Clear", command=self.clear)
        self.clear_button.pack(pady=2)

        self.help_text_id = None

    def toggle_waypoints(self):
        self.show_waypoints = not self.show_waypoints
        label = "Hide Waypoints" if self.show_waypoints else "Show Waypoints"
        self.toggle_waypoint_btn.config(text=label)
        self.redraw()

    def on_drag_start(self, event):
        for pt in self.points:
            x, y = pt.pos
            if abs(x - event.x) <= 6 and abs(y - event.y) <= 6:
                self.dragging_point = pt
                self.original_points = [p.pos for p in self.points]

                affected_segments = [
                    s for s in self.segments if s.p1 == pt.id or s.p2 == pt.id
                ]
                print(
                    f"🟡 Dragging agent point {pt.id}. Affected segments: {[s.id for s in affected_segments]}"
                )
                for seg in affected_segments:
                    line = LineString(
                        [self.points[seg.p1].pos, self.points[seg.p2].pos]
                    )
                    crossing_ids = []
                    for other in self.segments:
                        if seg.id != other.id:
                            other_line = LineString(
                                [self.points[other.p1].pos, self.points[other.p2].pos]
                            )
                            if line.crosses(other_line):
                                crossing_ids.append(other.id)
                    print(f"  ↳ Segment {seg.id} is crossing segments: {crossing_ids}")
                return

    def on_drag_motion(self, event):
        if self.dragging_point:
            self.dragging_point.pos = (event.x, event.y)
            self.redraw()

    def on_drag_end(self, event):
        if self.dragging_point:
            idx = self.dragging_point.id
            new_pos = (event.x, event.y)
            old_points = [
                KnotPoint(p.id, *pos, p.is_agent)
                for p, pos in zip(self.points, self.original_points)
            ]
            affected_segments = [s for s in self.segments if s.p1 == idx or s.p2 == idx]

            old_crossings = {}
            for seg in affected_segments:
                line = LineString([old_points[seg.p1].pos, old_points[seg.p2].pos])
                cross_ids = set()
                for other in self.segments:
                    if seg.id != other.id:
                        other_line = LineString(
                            [old_points[other.p1].pos, old_points[other.p2].pos]
                        )
                        if line.crosses(other_line):
                            cross_ids.add(other.id)
                old_crossings[seg.id] = cross_ids

            self.points[idx].pos = new_pos

            failed = False
            for seg in affected_segments:
                line = LineString([self.points[seg.p1].pos, self.points[seg.p2].pos])
                new_cross_ids = set()
                for other in self.segments:
                    if seg.id != other.id:
                        other_line = LineString(
                            [self.points[other.p1].pos, self.points[other.p2].pos]
                        )
                        if line.crosses(other_line):
                            new_cross_ids.add(other.id)
                print(
                    f"  ↳ Segment {seg.id} now crossing segments: {sorted(new_cross_ids)}"
                )
                if new_cross_ids != old_crossings[seg.id]:
                    added = new_cross_ids - old_crossings[seg.id]
                    removed = old_crossings[seg.id] - new_cross_ids
                    print(
                        f"❌ Crossing mismatch for segment {seg.id}. Added: {added}, Removed: {removed}"
                    )
                    failed = True

            if failed:
                print("❌ Constraint violation detected: restoring original position.")
                for i, pos in enumerate(self.original_points):
                    self.points[i].pos = pos
            else:
                print("✅ Constraint check passed: updating gaps.")
                check_preserve_crossings_and_update_gaps(
                    self.points, self.segments, idx
                )

            self.redraw()
            self.dragging_point = None

    def add_point(self, is_agent, is_underpass, x, y):
        new_point = KnotPoint(self.next_point_id, x, y, is_agent)
        self.next_point_id += 1
        self.points.append(new_point)

        if len(self.points) > 1:
            last_point = self.points[-2]
            self.add_segment(last_point, new_point, is_overpass=not is_underpass)

    def add_segment(self, p1_obj, p2_obj, is_overpass):
        new_line = LineString([p1_obj.pos, p2_obj.pos])
        new_seg = KnotSegment(self.next_segment_id, p1_obj.id, p2_obj.id, is_overpass)
        self.next_segment_id += 1

        for seg in self.segments:
            try:
                existing_p1 = self.get_point_by_id(seg.p1).pos
                existing_p2 = self.get_point_by_id(seg.p2).pos
            except ValueError:
                continue

            existing_line = LineString([existing_p1, existing_p2])
            if new_line.crosses(existing_line):
                pt = new_line.intersection(existing_line)
                if isinstance(pt, Point):
                    if is_overpass:
                        seg.gap_at.append(pt)
                    else:
                        new_seg.gap_at.append(pt)

        self.segments.append(new_seg)
        self.redraw()

    def get_point_by_id(self, point_id):
        for p in self.points:
            if p.id == point_id:
                return p
        raise ValueError(f"Point ID {point_id} not found")

    def redraw(self):
        self.canvas.delete("all")
        for seg in self.segments:
            p1 = self.get_point_by_id(seg.p1).pos
            p2 = self.get_point_by_id(seg.p2).pos
            if seg.gap_at:
                self.draw_with_gaps(p1, p2, seg.gap_at)
            else:
                self.canvas.create_line(
                    *p1,
                    *p2,
                    fill="black",
                    width=2,
                    arrow=tk.LAST,
                    arrowshape=(8, 10, 4),
                )

        for pt in self.points:
            x, y = pt.pos
            if pt.is_agent or self.show_waypoints:
                fill = "red" if pt.is_agent else "black"
                self.canvas.create_oval(
                    x - 5, y - 5, x + 5, y + 5, fill=fill, outline=""
                )

        self.draw_ui()

    def draw_with_gaps(self, p1, p2, gap_points, gap_size=15):
        line = LineString([p1, p2])
        length = line.length
        distances = [line.project(pt) for pt in gap_points]
        gaps = [(d - gap_size / 2, d + gap_size / 2) for d in distances]

        clipped = []
        for start, end in gaps:
            start = max(start, 0)
            end = min(end, length)
            if start < end:
                clipped.append((start, end))
        clipped.sort()

        draw_segments = []
        cursor = 0.0
        for start, end in clipped:
            if start > cursor:
                draw_segments.append((cursor, start))
            cursor = max(cursor, end)
        if cursor < length:
            draw_segments.append((cursor, length))

        for i, (start, end) in enumerate(draw_segments):
            pt1 = line.interpolate(start)
            pt2 = line.interpolate(end)
            is_last = i == len(draw_segments) - 1
            self.canvas.create_line(
                pt1.x,
                pt1.y,
                pt2.x,
                pt2.y,
                fill="black",
                width=2,
                arrow=tk.LAST if is_last else None,
                arrowshape=(8, 10, 4) if is_last else None,
            )

    def draw_ui(self):
        if self.help_text_id:
            self.canvas.delete(self.help_text_id)
        help_text = (
            "Instructions:\n"
            "- Drag points to move\n"
            "- Toggle waypoints to hide/show\n"
            "- Click Clear to reset"
        )
        self.help_text_id = self.canvas.create_text(
            10, 10, anchor="nw", text=help_text, font=("Arial", 9), fill="gray20"
        )

    def draw_sections(self, section_list, agent_points_set):
        self.clear()
        id_to_point = {}

        # First pass: gather all unique points
        raw_points = []
        for sec in section_list:
            for pt in [sec.start, sec.end]:
                if pt not in raw_points:
                    raw_points.append(pt)

        # Compute bounding box
        xs = [p[1] for p in raw_points]
        ys = [p[0] for p in raw_points]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)

        # Compute translation to center
        scale = 40
        padding = 20
        width, height = 600, 600  # canvas size
        knot_width = (max_x - min_x + 1) * scale
        knot_height = (max_y - min_y + 1) * scale
        offset_x = (width - knot_width) // 2 - min_x * scale + padding
        offset_y = (height - knot_height) // 2 - min_y * scale + padding

        # Create points
        for pt in raw_points:
            node_id = len(id_to_point)
            x = pt[1] * scale + offset_x
            y = pt[0] * scale + offset_y
            is_agent = pt in agent_points_set
            new_point = KnotPoint(node_id, x, y, is_agent)
            self.points.append(new_point)
            id_to_point[pt] = new_point

        # Create segments
        for sec in section_list:
            p1 = id_to_point[sec.start]
            p2 = id_to_point[sec.end]
            is_overpass = sec.over_under == 1
            self.add_segment(p1, p2, is_overpass)

        self.redraw()

    def clear(self):
        self.canvas.delete("all")
        self.points.clear()
        self.segments.clear()
        self.next_point_id = 0
        self.next_segment_id = 0
        self.redraw()


if __name__ == "__main__":
    root = tk.Tk()
    root.title("Knot Drawer")
    app = ShapelyGUI(root)
    app.draw_ui()
    root.mainloop()
