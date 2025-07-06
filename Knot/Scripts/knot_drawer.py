import tkinter as tk
from shapely.geometry import LineString, Point
import numpy as np
from geometry_utils import check_preserve_crossings_and_update_gaps
from region_detection import compute_agent_reduction, knot_manager, read_path
from shapely.ops import nearest_points

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
def compute_crossings_from_points(points, segments):
    crossings = {}
    for i, seg1 in enumerate(segments):
        line1 = LineString([points[seg1.p1].pos, points[seg1.p2].pos])
        for j in range(i + 1, len(segments)):
            seg2 = segments[j]
            line2 = LineString([points[seg2.p1].pos, points[seg2.p2].pos])
            if line1.crosses(line2):
                pt = line1.intersection(line2)
                if pt.geom_type == "Point":
                    crossings[frozenset([seg1.id, seg2.id])] = pt
    return crossings
def check_crossing_structure_equivalence(points, segments, initial_crossings):
    """
    Only checks that the same segment pairs cross — ignores crossing position.
    """
    def compute_crossing_pairs(ps):
        pairs = set()
        for i, seg1 in enumerate(segments):
            line1 = LineString([ps[seg1.p1].pos, ps[seg1.p2].pos])
            for j in range(i + 1, len(segments)):
                seg2 = segments[j]
                line2 = LineString([ps[seg2.p1].pos, ps[seg2.p2].pos])
                if line1.crosses(line2):
                    pairs.add(frozenset([seg1.id, seg2.id]))
        return pairs

    current_pairs = compute_crossing_pairs(points)
    original_pairs = set(initial_crossings.keys())

    if current_pairs != original_pairs:
        return False, "❌ Segment crossing pairs changed."

    return True, "✅ Segment pairs preserved."




def update_gaps_and_crossings(points, segments):
    def seg_line(seg):
        return LineString([points[seg.p1].pos, points[seg.p2].pos])
    for seg in segments: seg.gap_at.clear()
    for i in range(len(segments)):
        l1 = seg_line(segments[i])
        for j in range(i+1, len(segments)):
            l2 = seg_line(segments[j])
            if l1.crosses(l2):
                pt = l1.intersection(l2)
                if pt.geom_type=="Point":
                    if segments[i].is_overpass:
                        segments[j].gap_at.append(pt)
                    else:
                        segments[i].gap_at.append(pt)
class ShapelyGUI:
    def __init__(self, parent):
        self.canvas = tk.Canvas(parent, width=600, height=600, bg="white")
        self.canvas.pack()
        self.points, self.segments = [], []
        self.initial_crossings = {}
        self.next_point_id = self.next_segment_id = 0
        self.dragging_point = None
        self.original_positions = []
        self.repel_entry = tk.Entry(parent); self.repel_entry.insert(0,"0.5"); self.repel_entry.pack()
        self.attract_entry = tk.Entry(parent); self.attract_entry.insert(0,"0.05"); self.attract_entry.pack()
        self.toggle_btn = tk.Button(parent,text="Toggle Waypoints",command=self.redraw); self.toggle_btn.pack()
        self.next_btn = tk.Button(parent,text="Next Segment",command=self.next_segment); self.next_btn.pack()
        self.radius = 50
        self.locked_indices = set()
        self.straighten_step = 0
        self.ordered_indices = []
        self.canvas.bind("<ButtonPress-1>", self.on_drag_start)
        self.canvas.bind("<B1-Motion>", self.on_drag_motion)
        self.canvas.bind("<ButtonRelease-1>", self.on_drag_end)
        self.run_physics()

    def on_drag_start(self, e):
        for pt in self.points:
            x,y=pt.pos
            if abs(x-e.x)<=6 and abs(y-e.y)<=6:
                self.dragging_point=pt
                self.original_positions=[p.pos for p in self.points]
                break

    def on_drag_motion(self, e):
        if self.dragging_point:
            self.points[self.dragging_point.id].pos = (e.x, e.y)
            self.redraw()

    def on_drag_end(self, e):
        if not self.dragging_point:
            return
        idx = self.dragging_point.id
        old_pos = self.original_positions[idx]
        self.points[idx].pos = (e.x, e.y)
        ok, msg = check_crossing_structure_equivalence(self.points, self.segments, self.initial_crossings)

        if not ok:
            print(f"❌ Reverting drag of point {idx}: {msg}")
            self.points[idx].pos = old_pos
        else:
            print(f"✅ Drag complete for point {idx}")
        self.dragging_point = None
        self.redraw()

    def run_physics(self):
        try:
            rf = float(self.repel_entry.get())
            af = float(self.attract_entry.get())
        except:
            return

        for i, pt in enumerate(self.points):
            if i in self.locked_indices or pt == self.dragging_point:
                continue

            fx = fy = 0
            for j, other in enumerate(self.points):
                if i == j or j in self.locked_indices:
                    continue
                dx = other.pos[0] - pt.pos[0]
                dy = other.pos[1] - pt.pos[1]
                d = (dx * dx + dy * dy) ** 0.5
                if d == 0:
                    continue
                ux, uy = dx / d, dy / d
                f = -rf * (self.radius - d) if d < self.radius else af * (d - self.radius)
                fx += f * ux
                fy += f * uy

            old = pt.pos
            pt.pos = (pt.pos[0] + fx, pt.pos[1] + fy)
            ok, _ = check_crossing_structure_equivalence(self.points, self.segments, self.initial_crossings)
            if not ok:
                pt.pos = old

        self.redraw()
        self.canvas.after(50, self.run_physics)

    def next_segment(self):
        if self.straighten_step + 1 >= len(self.ordered_indices):
            return

        i1 = self.ordered_indices[self.straighten_step]
        i2 = self.ordered_indices[self.straighten_step + 1]
        print(f"🔒 Locking segment between agent {i1} and {i2}")
        print(f"   Anchor start: Point {i1} at {self.points[i1].pos}")
        print(f"   Anchor end:   Point {i2} at {self.points[i2].pos}")

        start, end = min(i1, i2), max(i1, i2)
        intermediates = [i for i in range(start + 1, end) if i not in self.locked_indices]

        print(f"   Locking intermediate points: {intermediates}")

        # Lock all points in this segment including intermediates
        self.locked_indices.update(intermediates + [i1, i2])

        self.straighten_step += 1
        self.redraw()

    def redraw(self):
        self.canvas.delete("all")
        curr = set()
        if self.straighten_step+1 < len(self.ordered_indices):
            curr = {self.ordered_indices[self.straighten_step], self.ordered_indices[self.straighten_step+1]}
        for seg in self.segments:
            p1=self.points[seg.p1].pos; p2=self.points[seg.p2].pos
            self.canvas.create_line(*p1,*p2,fill="black",width=2)
        for pt in self.points:
            x,y=pt.pos
            if pt.id in self.locked_indices:
                self.canvas.create_oval(x-6,y-6,x+6,y+6,fill="blue")
            elif pt.id in curr:
                self.canvas.create_oval(x-6,y-6,x+6,y+6,fill="gold")
            elif pt.is_agent:
                self.canvas.create_oval(x-6,y-6,x+6,y+6,fill="white",outline="red",width=2)
            else:
                self.canvas.create_oval(x-4,y-4,x+4,y+4,fill="black")

    def draw_sections(self, section_list, agent_points_set):
        self.clear()
        raw = []
        for sec in section_list:
            raw.append(sec.start)
            raw.append(sec.end)
        path_order = [section_list[0].start] + [sec.end for sec in section_list]
        print("Received Points in path order:")
        for p in path_order:
            print(f"   {'Agent' if p in agent_points_set else 'Turn'}@{p}")

        xs = [p[1] for p in path_order]
        ys = [p[0] for p in path_order]
        scale = 40
        ox = 50 - min(xs) * scale
        oy = 50 - min(ys) * scale
        id_map = {}

        for p in path_order:
            x, y = p[1] * scale + ox, p[0] * scale + oy
            idx = len(self.points)
            id_map[p] = idx
            self.points.append(KnotPoint(idx, x, y, p in agent_points_set))

        for sec in section_list:
            p1 = id_map[sec.start]
            p2 = id_map[sec.end]
            self.segments.append(KnotSegment(self.next_segment_id, p1, p2, sec.over_under == 1))
            self.next_segment_id += 1

        self.ordered_indices = [id_map[p] for p in path_order if p in agent_points_set]
        if len(self.ordered_indices) >= 2:
            self.locked_indices.update({self.ordered_indices[0], self.ordered_indices[1]})

        # Compute and store original crossing structure
        self.initial_crossings = compute_crossings_from_points(self.points, self.segments)

        self.redraw()

    def clear(self):
        self.canvas.delete("all")
        self.points = []
        self.segments = []
        self.next_point_id = self.next_segment_id = 0
        self.straighten_step = 0
        self.ordered_indices = []
        self.locked_indices.clear()


if __name__=="__main__":
    root=tk.Tk(); app=ShapelyGUI(root)
    try:
        mat, entry, exit = read_path()
        _,_,_,_,_,secs = compute_agent_reduction(mat,entry,exit)
        agents={p.pos_2d() for p in knot_manager.agent_registry.values()}
        app.draw_sections(secs,agents)
    except Exception as e:
        print("⚠️",e)
    root.mainloop()
