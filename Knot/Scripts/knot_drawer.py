import tkinter as tk
from shapely.geometry import LineString
import numpy as np
from geometry_utils import check_preserve_crossings_and_update_gaps
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


class ShapelyGUI:
    def __init__(self, parent):
        self.canvas = tk.Canvas(parent, width=600, height=600, bg="white")
        self.canvas.pack()

        self.points, self.segments = [], []
        self.initial_crossings = {}
        self.dragging_point = None
        self.original_positions = []
        self.velocities = []
        self.manual_locked_indices = set()
        self.locked_indices = set()
        self.equilibrium_distances = {}

        self.k_entry = tk.Entry(parent); self.k_entry.insert(0, "0.06"); self.k_entry.pack()
        self.c_entry = tk.Entry(parent); self.c_entry.insert(0, "0.2"); self.c_entry.pack()
        self.m_entry = tk.Entry(parent); self.m_entry.insert(0, "1.0"); self.m_entry.pack()
        self.dt_entry = tk.Entry(parent); self.dt_entry.insert(0, "0.2"); self.dt_entry.pack()

        self.set_btn = tk.Button(parent, text="Set Physics", command=self.update_physics_constants); self.set_btn.pack()
        self.toggle_btn = tk.Button(parent, text="Toggle Waypoints", command=self.redraw); self.toggle_btn.pack()
        self.next_btn = tk.Button(parent, text="Next Segment", command=self.next_segment); self.next_btn.pack()

        self.radius = 50
        self.straighten_step = 0
        self.ordered_indices = []

        self.canvas.bind("<ButtonPress-1>", self.on_drag_start)
        self.canvas.bind("<B1-Motion>", self.on_drag_motion)
        self.canvas.bind("<ButtonRelease-1>", self.on_drag_end)

        self.run_physics()

    def lock_points(self, point_ids):
        for pid in point_ids:
            self.manual_locked_indices.discard(pid)
            self.locked_indices.add(pid)
        print(f"🔒 Locked points: {point_ids}")

    def update_physics_constants(self):
        try:
            self.k = float(self.k_entry.get())
            self.c = float(self.c_entry.get())
            self.m = float(self.m_entry.get())
            self.dt = float(self.dt_entry.get())
            print(f"✅ Physics updated: k={self.k}, c={self.c}, m={self.m}, dt={self.dt}")
        except ValueError:
            print("⚠️ Invalid input.")

    def on_drag_start(self, e):
        for pt in self.points:
            if pt.id in self.locked_indices:
                continue
            x, y = pt.pos
            if abs(x - e.x) <= 6 and abs(y - e.y) <= 6:
                self.dragging_point = pt
                self.original_positions = [p.pos for p in self.points]
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
        k = getattr(self, "k", 0.06)
        c = getattr(self, "c", 0.2)
        m = getattr(self, "m", 1.0)
        dt = getattr(self, "dt", 0.2)

        try:
            for i, pti in enumerate(self.points):
                if pti == self.dragging_point or i in self.locked_indices or i in self.manual_locked_indices:
                    continue

                force = np.array([0.0, 0.0])
                for j, ptj in enumerate(self.points):
                    if i == j:
                        continue
                    dx = ptj.pos[0] - pti.pos[0]
                    dy = ptj.pos[1] - pti.pos[1]
                    disp = np.array([dx, dy])
                    dist = np.linalg.norm(disp)
                    if dist == 0:
                        continue
                    direction = disp / dist
                    key = tuple(sorted((i, j)))
                    L0 = self.equilibrium_distances.get(key, self.radius)
                    stretch = dist - L0
                    f_spring = k * stretch * direction
                    dv = self.velocities[j] - self.velocities[i]
                    v_rel = np.dot(dv, direction)
                    f_damp = c * v_rel * direction
                    force += f_spring + f_damp

                acc = force / m
                self.velocities[i] += acc * dt
                disp = self.velocities[i] * dt
                if np.linalg.norm(disp) > 5.0:
                    disp = disp / np.linalg.norm(disp) * 5.0
                new_pos = np.array(pti.pos) + disp
                old_pos = pti.pos
                pti.pos = tuple(new_pos)
                ok, _ = check_crossing_structure_equivalence(self.points, self.segments, self.initial_crossings)
                if not ok:
                    pti.pos = old_pos
                    self.velocities[i] = np.array([0.0, 0.0])
        except Exception as e:
            print(f"⚠️ Error in physics loop: {e}")

        self.redraw()
        self.canvas.after(20, self.run_physics)

    def next_segment(self):
        if self.straighten_step + 1 >= len(self.ordered_indices):
            return

        # 1. Promote manual-locked to fully locked
        for pid in self.manual_locked_indices.copy():
            self.locked_indices.add(pid)
        self.manual_locked_indices.clear()

        # 2. Get current segment (i1 → i2)
        i1 = self.ordered_indices[self.straighten_step]
        i2 = self.ordered_indices[self.straighten_step + 1]
        print(f"🔒 Locking segment between agent {i1} and {i2}")

        path_ids = [p.id for p in self.points]
        start_idx = path_ids.index(i1)
        end_idx = path_ids.index(i2)
        if start_idx > end_idx:
            start_idx, end_idx = end_idx, start_idx

        intermediates = [self.points[i].id for i in range(start_idx + 1, end_idx)]

        # 3. LOCK current segment
        to_lock = [i1, i2] + intermediates

        # 4. LOCK the next segment’s END agent (i3), if it exists
        if self.straighten_step + 2 < len(self.ordered_indices):
            i3 = self.ordered_indices[self.straighten_step + 2]
            to_lock.append(i3)

            # Pre-lock the intermediates of next segment (as manual)
            start_idx2 = path_ids.index(i2)
            end_idx2 = path_ids.index(i3)
            if start_idx2 > end_idx2:
                start_idx2, end_idx2 = end_idx2, start_idx2
            for i in range(start_idx2 + 1, end_idx2):
                self.manual_locked_indices.add(self.points[i].id)

        self.lock_points(to_lock)

        self.straighten_step += 1
        self.redraw()

    def redraw(self):
        self.canvas.delete("all")
        curr = set()
        if self.straighten_step + 1 < len(self.ordered_indices):
            curr = {self.ordered_indices[self.straighten_step], self.ordered_indices[self.straighten_step + 1]}

        for seg in self.segments:
            p1, p2 = self.points[seg.p1].pos, self.points[seg.p2].pos
            self.canvas.create_line(*p1, *p2, fill="black", width=2)
        for pt in self.points:
            x, y = pt.pos
            pid = pt.id

            if pid in self.locked_indices:
                if pt.is_agent:
                    self.canvas.create_oval(x - 6, y - 6, x + 6, y + 6, fill="blue4")  # dark blue
                else:
                    self.canvas.create_oval(x - 6, y - 6, x + 6, y + 6, fill="cornflowerblue")  # light blue
            elif pid in self.manual_locked_indices:
                self.canvas.create_oval(x - 6, y - 6, x + 6, y + 6, fill="skyblue")
            elif pid in curr and pid not in self.locked_indices:
                self.canvas.create_oval(x - 6, y - 6, x + 6, y + 6, fill="gold")
            elif pt.is_agent:
                self.canvas.create_oval(x - 6, y - 6, x + 6, y + 6, fill="white", outline="red", width=2)
            else:
                self.canvas.create_oval(x - 4, y - 4, x + 4, y + 4, fill="black")

    def draw_sections(self, section_list, agent_points_set):
        self.clear()
        path_order = [section_list[0].start] + [sec.end for sec in section_list]
        xs = [p[1] for p in path_order]
        ys = [p[0] for p in path_order]
        scale, ox, oy = 40, 50 - min(xs) * 40, 50 - min(ys) * 40
        id_map = {}
        for p in path_order:
            x, y = p[1] * scale + ox, p[0] * scale + oy
            idx = len(self.points)
            id_map[p] = idx
            self.points.append(KnotPoint(idx, x, y, p in agent_points_set))
        for sec in section_list:
            p1, p2 = id_map[sec.start], id_map[sec.end]
            self.segments.append(KnotSegment(len(self.segments), p1, p2, sec.over_under == 1))
        self.ordered_indices = [id_map[p] for p in path_order if p in agent_points_set]
        if len(self.ordered_indices) >= 2:
            i1, i2 = self.ordered_indices[0], self.ordered_indices[1]
            path_ids = [p.id for p in self.points]
            start_idx = path_ids.index(i1)
            end_idx = path_ids.index(i2)
            if start_idx > end_idx:
                start_idx, end_idx = end_idx, start_idx
            for i in range(start_idx + 1, end_idx):
                self.manual_locked_indices.add(self.points[i].id)
            self.locked_indices.update({i1, i2})
        self.initial_crossings = compute_crossings_from_points(self.points, self.segments)
        self.velocities = [np.array([0.0, 0.0]) for _ in self.points]
        self.equilibrium_distances = {}
        for i, pti in enumerate(self.points):
            for j, ptj in enumerate(self.points):
                if i < j:
                    d = np.hypot(ptj.pos[0] - pti.pos[0], ptj.pos[1] - pti.pos[1])
                    self.equilibrium_distances[(i, j)] = d
        self.redraw()

    def clear(self):
        self.canvas.delete("all")
        self.points = []
        self.segments = []
        self.velocities = []
        self.next_point_id = self.next_segment_id = 0
        self.straighten_step = 0
        self.ordered_indices = []
        self.locked_indices.clear()
        self.manual_locked_indices.clear()


if __name__ == "__main__":
    root = tk.Tk()
    app = ShapelyGUI(root)
    try:
        mat, entry, exit = read_path()
        _, _, _, _, _, secs = compute_agent_reduction(mat, entry, exit)
        agents = {p.pos_2d() for p in knot_manager.agent_registry.values()}
        app.draw_sections(secs, agents)
    except Exception as e:
        print("⚠️", e)
    root.mainloop()
