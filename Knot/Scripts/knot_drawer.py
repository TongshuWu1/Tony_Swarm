import tkinter as tk
import math
from shapely.geometry import LineString, Point
from region_detection import compute_agent_reduction, knot_manager

class KnotPoint:
    def __init__(self, point_id, x, y, is_agent):
        self.id = point_id
        self.pos = (x, y)
        self.is_agent = is_agent
        self.locked = is_agent

class KnotSegment:
    def __init__(self, seg_id, p1_id, p2_id, is_overpass):
        self.id = seg_id
        self.p1 = p1_id
        self.p2 = p2_id
        self.is_overpass = is_overpass
        self.gap_at = []


def update_gaps_and_crossings(points, segments):
    for seg in segments:
        seg.gap_at.clear()
    segment_layers = {}
    def seg_line(seg):
        return LineString([points[seg.p1].pos, points[seg.p2].pos])
    for i, seg1 in enumerate(segments):
        line1 = seg_line(seg1)
        for j in range(i + 1, len(segments)):
            seg2 = segments[j]
            line2 = seg_line(seg2)
            if line1.crosses(line2):
                pt = line1.intersection(line2)
                if pt.geom_type != "Point":
                    continue
                if seg1.is_overpass and not seg2.is_overpass:
                    seg2.gap_at.append((pt, seg1.id))
                elif seg2.is_overpass and not seg1.is_overpass:
                    seg1.gap_at.append((pt, seg2.id))
                else:
                    seg2.gap_at.append((pt, seg1.id))
    return segment_layers


def generate_segment_with_turns(p1, p2, is_overpass, points, segments):
    new_points = []
    new_segments = []
    line = LineString([p1.pos, p2.pos])
    existing_lines = [(seg, LineString([points[seg.p1].pos, points[seg.p2].pos])) for seg in segments]
    crossings = []
    for seg, other_line in existing_lines:
        if line.crosses(other_line):
            pt = line.intersection(other_line)
            if pt.geom_type == "Point":
                crossings.append((pt, seg))

    crossings.sort(key=lambda c: line.project(c[0]))
    current_point = p1
    for pt, crossed_seg in crossings:
        dx = pt.x - current_point.pos[0]
        dy = pt.y - current_point.pos[1]
        norm = math.hypot(dx, dy)
        if norm == 0:
            continue
        offset = 10
        tx = current_point.pos[0] + dx / norm * (norm - offset)
        ty = current_point.pos[1] + dy / norm * (norm - offset)
        turn_pt = KnotPoint(len(points) + len(new_points), tx, ty, False)
        new_points.append(turn_pt)
        new_segments.append(KnotSegment(len(segments) + len(new_segments), current_point.id, turn_pt.id, is_overpass))
        current_point = turn_pt

    new_segments.append(KnotSegment(len(segments) + len(new_segments), current_point.id, p2.id, is_overpass))
    return new_points, new_segments


class ShapelyGUI:
    def __init__(self, parent):
        self.parent = parent
        self.canvas = tk.Canvas(parent, width=600, height=600, bg="white")
        self.canvas.pack()

        self.points = []
        self.segments = []
        self.point_velocities = {}
        self.relax_in_progress = False
        self.relax_pair = None
        self.initial_topology = {}
        self.previous_positions = []
        self.remaining_sections = []
        self.agent_points_set = set()
        self.id_map = {}

        button_frame = tk.Frame(parent)
        button_frame.pack()
        tk.Button(button_frame, text="Clear", command=self.clear).pack(side=tk.LEFT)
        tk.Button(button_frame, text="Generate", command=self.initialize_drawing).pack(side=tk.LEFT)
        tk.Button(button_frame, text="Next Step", command=self.step).pack(side=tk.LEFT)

        self.parent.after(20, self.relax_loop)

    def clear(self):
        self.points.clear()
        self.segments.clear()
        self.remaining_sections.clear()
        self.id_map.clear()
        self.canvas.delete("all")
        self.redraw()

    def add_point(self, x, y, is_agent):
        pt = KnotPoint(len(self.points), x, y, is_agent)
        self.points.append(pt)
        return pt

    def add_segment(self, p1, p2, is_overpass):
        seg = KnotSegment(len(self.segments), p1.id, p2.id, is_overpass)
        self.segments.append(seg)
        update_gaps_and_crossings(self.points, self.segments)

    def redraw(self):
        self.canvas.delete("all")
        for seg in self.segments:
            p1 = self.points[seg.p1].pos
            p2 = self.points[seg.p2].pos
            if seg.gap_at:
                self.draw_with_gaps(p1, p2, seg.gap_at)
            else:
                self.draw_segment(p1, p2)
        for pt in self.points:
            x, y = pt.pos
            color = "white" if pt.is_agent else "black"
            outline = "red" if pt.is_agent else ""
            self.canvas.create_oval(x - 5, y - 5, x + 5, y + 5, fill=color, outline=outline)

    def draw_segment(self, p1, p2):
        self.canvas.create_line(*p1, *p2, fill="black", width=2)

    def draw_with_gaps(self, p1, p2, gaps, gap_size=15):
        line = LineString([p1, p2])
        length = line.length
        ranges = [(line.project(pt) - gap_size / 2, line.project(pt) + gap_size / 2) for pt, _ in gaps]
        draw_ranges = []
        cursor = 0.0
        for start, end in sorted(ranges):
            if start > cursor:
                draw_ranges.append((cursor, start))
            cursor = max(cursor, end)
        if cursor < length:
            draw_ranges.append((cursor, length))
        for start, end in draw_ranges:
            pt1 = line.interpolate(start)
            pt2 = line.interpolate(end)
            self.canvas.create_line(pt1.x, pt1.y, pt2.x, pt2.y, fill="black", width=2)

    def initialize_drawing(self):
        self.clear()
        try:
            matrix = knot_manager.matrix
            entry = knot_manager.entry
            exit = knot_manager.exit
            _, _, _, _, _, section_list = compute_agent_reduction(matrix, entry, exit)
            self.agent_points_set = {p.pos_2d() for p in knot_manager.agent_registry.values()}
            raw_pts = list({pt for sec in section_list for pt in [sec.start, sec.end]})
            scale = 40
            offset_x = 100
            offset_y = 100
            for pt in raw_pts:
                x, y = pt[1] * scale + offset_x, pt[0] * scale + offset_y
                new_pt = self.add_point(x, y, pt in self.agent_points_set)
                self.id_map[pt] = new_pt
            self.remaining_sections = section_list
            self.step()
        except Exception as e:
            import traceback
            print("Initialization failed:", e)
            traceback.print_exc()

    def step(self):
        if self.relax_in_progress:
            return

        if not self.remaining_sections:
            print("All segments added.")
            return

        sec = self.remaining_sections.pop(0)
        p1 = self.id_map[sec.start]
        p2 = self.id_map[sec.end]
        is_over = sec.over_under == 1

        if p1.is_agent and p2.is_agent:
            new_pts, new_segs = generate_segment_with_turns(p1, p2, is_over, self.points, self.segments)
            for pt in new_pts:
                self.points.append(pt)
            for seg in new_segs:
                self.segments.append(seg)
            update_gaps_and_crossings(self.points, self.segments)
            self.initial_topology = update_gaps_and_crossings(self.points, self.segments)
            self.previous_positions = [pt.pos for pt in self.points]
            self.relax_in_progress = True
            self.relax_pair = (p1, p2)
        else:
            self.add_segment(p1, p2, is_over)
            self.redraw()

    def relax_loop(self):
        self.relax_step()
        self.parent.after(20, self.relax_loop)

    def relax_step(self, dt=1.0, damping=0.8):
        if not self.relax_in_progress:
            return

        forces = {pt.id: [0.0, 0.0] for pt in self.points if not pt.locked}
        neighbor_map = {}
        for seg in self.segments:
            neighbor_map.setdefault(seg.p1, []).append(seg.p2)
            neighbor_map.setdefault(seg.p2, []).append(seg.p1)
        for pid, pt in enumerate(self.points):
            if pt.locked:
                continue
            for nbr_id in neighbor_map.get(pid, []):
                nbr = self.points[nbr_id]
                dx = nbr.pos[0] - pt.pos[0]
                dy = nbr.pos[1] - pt.pos[1]
                forces[pid][0] += dx * 0.1
                forces[pid][1] += dy * 0.1
        still_moving = False
        for pid, pt in enumerate(self.points):
            if pt.locked:
                continue
            fx, fy = forces[pid]
            vx, vy = self.point_velocities.get(pid, [0.0, 0.0])
            vx = (vx + fx * dt) * damping
            vy = (vy + fy * dt) * damping
            if abs(vx) > 0.01 or abs(vy) > 0.01:
                still_moving = True
            self.point_velocities[pid] = [vx, vy]
            pt.pos = (pt.pos[0] + vx * dt, pt.pos[1] + vy * dt)

        if not still_moving:
            self.relax_in_progress = False
            if self.relax_pair:
                for pt in self.relax_pair:
                    pt.locked = True
            self.relax_pair = None

        self.redraw()


if __name__ == "__main__":
    root = tk.Tk()
    root.title("Knot Builder")
    app = ShapelyGUI(root)
    root.mainloop()
