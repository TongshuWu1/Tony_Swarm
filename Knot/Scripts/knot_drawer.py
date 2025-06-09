# knot_drawer.py

import tkinter as tk
from shapely.geometry import LineString, Point
from region_detection import compute_agent_reduction, knot_manager, read_path


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


def check_crossing_structure_equivalence(points, segments, moved_idx):
    def seg_line(seg):
        return LineString([points[seg.p1].pos, points[seg.p2].pos])

    def section_id(a, b):
        return tuple(sorted((a, b)))

    original_crosses = set()
    new_crosses = set()
    moved_segments = [s for s in segments if s.p1 == moved_idx or s.p2 == moved_idx]

    for s1 in moved_segments:
        l1 = seg_line(s1)
        for s2 in segments:
            if s1.id == s2.id:
                continue
            l2 = seg_line(s2)
            if l1.crosses(l2):
                pair = tuple(sorted((s1.id, s2.id)))
                original_crosses.add(pair)

    for s1 in moved_segments:
        l1 = seg_line(s1)
        for s2 in segments:
            if s1.id == s2.id:
                continue
            l2 = seg_line(s2)
            if l1.crosses(l2):
                pair = tuple(sorted((s1.id, s2.id)))
                new_crosses.add(pair)

    return original_crosses == new_crosses


def update_gaps_and_crossings(points, segments):
    def seg_line(seg):
        return LineString([points[seg.p1].pos, points[seg.p2].pos])

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
        self.original_positions = []

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
                self.original_positions = [p.pos for p in self.points]
                return

    def on_drag_motion(self, event):
        if self.dragging_point:
            self.dragging_point.pos = (event.x, event.y)
            self.redraw()

    def on_drag_end(self, event):
        if not self.dragging_point:
            return

        idx = self.dragging_point.id
        self.points[idx].pos = (event.x, event.y)

        if check_crossing_structure_equivalence(self.points, self.segments, idx):
            update_gaps_and_crossings(self.points, self.segments)
            print("✅ Valid move: crossing structure preserved or Reidemeister-compatible.")
        else:
            print("❌ Invalid move: crossing structure violated. Reverting.")
            for i, pos in enumerate(self.original_positions):
                self.points[i].pos = pos

        self.redraw()
        self.dragging_point = None

    def add_point(self, is_agent, is_underpass, x, y):
        new_point = KnotPoint(self.next_point_id, x, y, is_agent)
        self.points.append(new_point)
        self.next_point_id += 1

        if len(self.points) > 1:
            last_point = self.points[-2]
            self.add_segment(last_point, new_point, is_overpass=not is_underpass)

    def add_segment(self, p1_obj, p2_obj, is_overpass):
        new_seg = KnotSegment(self.next_segment_id, p1_obj.id, p2_obj.id, is_overpass)
        self.next_segment_id += 1
        self.segments.append(new_seg)
        update_gaps_and_crossings(self.points, self.segments)
        self.redraw()

    def get_point_by_id(self, pid):
        return next(p for p in self.points if p.id == pid)

    def redraw(self):
        self.canvas.delete("all")
        for seg in self.segments:
            p1 = self.get_point_by_id(seg.p1).pos
            p2 = self.get_point_by_id(seg.p2).pos
            if seg.gap_at:
                self.draw_with_gaps(p1, p2, seg.gap_at)
            else:
                self.canvas.create_line(*p1, *p2, fill="black", width=2, arrow=tk.LAST)

        for pt in self.points:
            x, y = pt.pos
            if pt.is_agent:
                self.canvas.create_oval(
                    x - 6, y - 6, x + 6, y + 6, fill="white", outline="red", width=2
                )
            elif self.show_waypoints:
                self.canvas.create_oval(x - 4, y - 4, x + 4, y + 4, fill="black")

        self.draw_ui()

    def draw_with_gaps(self, p1, p2, gaps, gap_size=15):
        line = LineString([p1, p2])
        length = line.length
        ranges = [(line.project(g) - gap_size / 2, line.project(g) + gap_size / 2) for g in gaps]
        draw_ranges = []
        cursor = 0.0
        for start, end in sorted(ranges):
            if start > cursor:
                draw_ranges.append((cursor, start))
            cursor = max(cursor, end)
        if cursor < length:
            draw_ranges.append((cursor, length))

        for i, (start, end) in enumerate(draw_ranges):
            pt1, pt2 = line.interpolate(start), line.interpolate(end)
            self.canvas.create_line(
                pt1.x, pt1.y, pt2.x, pt2.y,
                fill="black", width=2,
                arrow=tk.LAST if i == len(draw_ranges) - 1 else None,
            )

    def draw_ui(self):
        if self.help_text_id:
            self.canvas.delete(self.help_text_id)
        msg = (
            "Instructions:\n"
            "- Drag points to move\n"
            "- Crossing structure is constrained\n"
            "- Click Clear to reset"
        )
        self.help_text_id = self.canvas.create_text(
            10, 10, anchor="nw", text=msg, font=("Arial", 9), fill="gray30"
        )

    def draw_sections(self, section_list, agent_points_set):
        self.clear()
        id_to_point = {}

        raw_points = list({pt for sec in section_list for pt in [sec.start, sec.end]})
        xs = [p[1] for p in raw_points]
        ys = [p[0] for p in raw_points]
        scale = 40
        offset_x = 50 - min(xs) * scale
        offset_y = 50 - min(ys) * scale

        for pt in raw_points:
            x, y = pt[1] * scale + offset_x, pt[0] * scale + offset_y
            new_pt = KnotPoint(len(self.points), x, y, pt in agent_points_set)
            id_to_point[pt] = new_pt
            self.points.append(new_pt)

        for sec in section_list:
            p1 = id_to_point[sec.start]
            p2 = id_to_point[sec.end]
            self.add_segment(p1, p2, is_overpass=sec.over_under == 1)

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

    try:
        matrix, entry, exit = read_path()
        _, _, _, _, _, sections = compute_agent_reduction(matrix, entry, exit)
        agents = {p.pos_2d() for p in knot_manager.agent_registry.values()}
        app.draw_sections(sections, agents)
    except Exception as e:
        print("⚠️ Error initializing:", e)

    root.mainloop()
