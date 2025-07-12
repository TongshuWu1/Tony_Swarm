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
        self.frozen_intermediates = set()
        self.equilibrium_distances = {}

        self.convergence_skip_frames =5
        self.frames_since_segment_start = 10

        self.converged_counter = 0
        self.convergence_threshold = 0.4
        self.converged_steps_required = 10

        self.avg_speed_buffer = []
        self.avg_speed_window_size = 7  # or 15

        self.prev_force_dirs = [np.zeros(2, dtype=float) for _ in range(len(self.points))]
        self.vibrate_count = [0 for _ in range(len(self.points))]

        param_frame = tk.Frame(parent)
        param_frame.pack()

        def add_param(row, label, entry_attr, default):
            tk.Label(param_frame, text=label).grid(row=row, column=0, sticky='w')
            entry = tk.Entry(param_frame, width=8)
            entry.insert(0, str(default))
            entry.grid(row=row, column=1)
            setattr(self, entry_attr, entry)

        add_param(0, "Spring (k)", "k_entry", 0.07)
        add_param(1, "Damping (c)", "c_entry", 0.08)
        add_param(2, "Mass (m)", "m_entry", 0.6)
        add_param(3, "Time Step (dt)", "dt_entry", 0.4)
        add_param(4, "Straighten", "straighten_force_entry", 1.3)
        add_param(5, "Repel Strength", "repulsion_entry", 4.0)
        add_param(6, "Min Dist", "min_dist_entry", 30.0)
        add_param(7, "Locked Mult", "locked_repel_multiplier_entry", 10.0)
        add_param(8, "Conv Thresh", "conv_thresh_entry", 0.12)
        add_param(9, "Conv Frames", "conv_steps_entry", 12)

        self.set_btn = tk.Button(parent, text="Set Physics", command=self.update_physics_constants)
        self.set_btn.pack()

        self.toggle_btn = tk.Button(parent, text="Toggle Waypoints", command=self.redraw)
        self.toggle_btn.pack()
        self.next_btn = tk.Button(parent, text="Next Segment", command=self.next_segment)
        self.next_btn.pack()

        self.radius = 50
        self.straighten_step = 0
        self.ordered_indices = []

        self.canvas.bind("<ButtonPress-1>", self.on_drag_start)
        self.canvas.bind("<B1-Motion>", self.on_drag_motion)
        self.canvas.bind("<ButtonRelease-1>", self.on_drag_end)

        self.update_physics_constants()
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
            self.straighten_force = float(self.straighten_force_entry.get())
            self.repulsion_strength = float(self.repulsion_entry.get())
            self.min_dist_threshold = float(self.min_dist_entry.get())
            self.locked_repel_multiplier = float(self.locked_repel_multiplier_entry.get())
            self.convergence_threshold = float(self.conv_thresh_entry.get())
            self.converged_steps_required = int(self.conv_steps_entry.get())

            print(
                f"✅ Physics updated: k={self.k}, c={self.c}, m={self.m}, dt={self.dt}, "
                f"straighten_force={self.straighten_force}, repulsion_strength={self.repulsion_strength}, "
                f"min_dist_threshold={self.min_dist_threshold}, "
                f"convergence_threshold={self.convergence_threshold}, "
                f"required stable frames={self.converged_steps_required}"
            )
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
        def normalize(vec):
            vec = np.array(vec, dtype=float)
            norm = np.linalg.norm(vec)
            if norm < 1e-8:
                return vec
            return vec / norm

        k = getattr(self, "k", 0.06)
        c = getattr(self, "c", 0.1)
        m = getattr(self, "m", 1.0)
        dt = getattr(self, "dt", 0.3)
        straighten_strength = getattr(self, "straighten_force", 1.4)
        repulsion_strength = getattr(self, "repulsion_strength", 2.0)
        min_dist_threshold = getattr(self, "min_dist_threshold", 10.0)
        locked_repel_multiplier = getattr(self, "locked_repel_multiplier", 5.0)
        max_velocity = 0.6

        try:
            self.frames_since_segment_start += 1

            active_intermediates = set()
            if self.straighten_step + 1 < len(self.ordered_indices):
                i1 = self.ordered_indices[self.straighten_step]
                i2 = self.ordered_indices[self.straighten_step + 1]
                path_ids = [p.id for p in self.points]
                start_idx = path_ids.index(i1)
                end_idx = path_ids.index(i2)
                if start_idx > end_idx:
                    start_idx, end_idx = end_idx, start_idx
                active_intermediates = set(path_ids[start_idx + 1:end_idx])

            force_map = {i: np.zeros(2, dtype=float) for i in range(len(self.points))}

            for i, pti in enumerate(self.points):
                for j, ptj in enumerate(self.points):
                    if i == j:
                        continue
                    dx = ptj.pos[0] - pti.pos[0]
                    dy = ptj.pos[1] - pti.pos[1]
                    disp = np.array([dx, dy], dtype=float)
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
                    f_total = f_spring + f_damp
                    if i not in self.locked_indices and i not in self.manual_locked_indices:
                        force_map[i] += f_total

            if self.straighten_step + 1 < len(self.ordered_indices):
                for i in range(start_idx + 1, end_idx):
                    pid = path_ids[i]
                    if pid not in self.manual_locked_indices:
                        continue
                    p_prev = np.array(self.points[path_ids[i - 1]].pos, dtype=float)
                    p_curr = np.array(self.points[pid].pos, dtype=float)
                    p_next = np.array(self.points[path_ids[i + 1]].pos, dtype=float)
                    v1 = normalize(p_prev - p_curr)
                    v2 = normalize(p_next - p_curr)
                    bisector = normalize(v1 + v2)
                    force_map[pid] += straighten_strength * bisector

            for i, pt in enumerate(self.points):
                pt_pos = np.array(pt.pos, dtype=float)
                for seg in self.segments:
                    if i == seg.p1 or i == seg.p2:
                        continue
                    a = np.array(self.points[seg.p1].pos, dtype=float)
                    b = np.array(self.points[seg.p2].pos, dtype=float)
                    ab = b - a
                    ab_len_sq = np.dot(ab, ab)
                    if ab_len_sq == 0:
                        continue
                    t = np.clip(np.dot(pt_pos - a, ab) / ab_len_sq, 0, 1)
                    closest = a + t * ab
                    disp = closest - pt_pos
                    dist = np.linalg.norm(disp)
                    if dist < 1e-5:
                        disp = np.random.randn(2) * 0.01
                        dist = 1e-5
                    if dist < min_dist_threshold:
                        repel_dir = normalize(disp)
                        delta = min((min_dist_threshold - dist) / min_dist_threshold, 0.5)
                        repel_mag = repulsion_strength * (
                            locked_repel_multiplier if i in self.locked_indices else 1.0) * (delta ** 2)
                        repel_force = repel_dir * repel_mag
                        force_map[i] -= repel_force
                        force_map[seg.p1] += 0.5 * repel_force
                        force_map[seg.p2] += 0.5 * repel_force

            for i in active_intermediates:
                f_now = force_map[i]
                if np.linalg.norm(f_now) > 5.0:
                    print(f"🧊 Force-freezing point {i} due to jitter spike")
                    self.velocities[i] = np.zeros(2)
                    self.frozen_intermediates.add(i)
                    continue
                self.prev_force_dirs[i] = f_now

            for i, pti in enumerate(self.points):
                if pti == self.dragging_point or i in self.locked_indices or i in self.frozen_intermediates:
                    continue
                acc = force_map[i] / m
                raw_velocity = acc * dt
                self.velocities[i] = 0.8 * self.velocities[i] + 0.2 * raw_velocity
                speed = np.linalg.norm(self.velocities[i])
                if speed > max_velocity:
                    self.velocities[i] = self.velocities[i] / speed * max_velocity
                disp = self.velocities[i] * dt
                new_pos = np.array(pti.pos, dtype=float) + disp
                old_pos = pti.pos
                pti.pos = tuple(new_pos)
                ok, _ = check_crossing_structure_equivalence(self.points, self.segments, self.initial_crossings)
                if not ok:
                    pti.pos = old_pos
                    self.velocities[i] = np.zeros(2)

            if self.frames_since_segment_start <= self.convergence_skip_frames:
                pass
                # print(
                #     f"⏳ Skipping convergence check ({self.frames_since_segment_start}/{self.convergence_skip_frames})")
            else:
                intermediates = [i for i in self.manual_locked_indices if i not in self.locked_indices]
                if not intermediates:
                    print("✅ No intermediates remaining. Advancing to next segment.")
                    self.next_segment()
                    self.converged_counter = 0
                    self.frames_since_segment_start = 0
                elif all(i in self.frozen_intermediates for i in intermediates):
                    print("✅ All intermediates frozen. Advancing to next segment.")
                    self.next_segment()
                    self.converged_counter = 0
                    self.frames_since_segment_start = 0
                else:
                    moving_speeds = [np.linalg.norm(self.velocities[i]) for i in intermediates if
                                     i not in self.frozen_intermediates]
                    if not hasattr(self, "avg_speed_buffer"):
                        self.avg_speed_buffer = []
                        self.avg_speed_window_size = 10
                    instant_speed = np.mean(moving_speeds) if moving_speeds else 0.0
                    if instant_speed > 1.0:
                        print(f"⚠️ Detected speed spike: {instant_speed:.4f} — purging buffer")
                        self.avg_speed_buffer.clear()
                        self.converged_counter = 0
                    else:
                        self.avg_speed_buffer.append(instant_speed)
                        if len(self.avg_speed_buffer) > self.avg_speed_window_size:
                            self.avg_speed_buffer.pop(0)
                        avg_speed = np.mean(self.avg_speed_buffer)
                        print(
                            f"🔍 Instant speed = {instant_speed:.5f} | Smoothed avg = {avg_speed:.5f} | Stable frames: {self.converged_counter}/{self.converged_steps_required}")
                        if avg_speed < self.convergence_threshold:
                            self.converged_counter += 1
                        else:
                            self.converged_counter = 0
                        if self.converged_counter >= self.converged_steps_required:
                            print("✅ Speed convergence reached. Advancing to next segment.")
                            self.next_segment()
                            self.converged_counter = 0
                            self.frames_since_segment_start = 0

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
        self.frames_since_segment_start = 0

    def redraw(self):
        self.canvas.delete("all")
        curr = set()
        if self.straighten_step + 1 < len(self.ordered_indices):
            curr = {self.ordered_indices[self.straighten_step], self.ordered_indices[self.straighten_step + 1]}
        elif self.straighten_step < len(self.ordered_indices):
            curr = {self.ordered_indices[self.straighten_step]}

        # === Draw segments with over/under gap handling ===
        drawn_pairs = set()
        already_drawn = set()

        # First draw all underpasses with gaps
        for i, seg1 in enumerate(self.segments):
            for j in range(i + 1, len(self.segments)):
                seg2 = self.segments[j]
                key = tuple(sorted((seg1.id, seg2.id)))
                if key in drawn_pairs:
                    continue
                drawn_pairs.add(key)

                p1a = np.array(self.points[seg1.p1].pos)
                p1b = np.array(self.points[seg1.p2].pos)
                p2a = np.array(self.points[seg2.p1].pos)
                p2b = np.array(self.points[seg2.p2].pos)

                line1 = LineString([p1a, p1b])
                line2 = LineString([p2a, p2b])

                if not line1.crosses(line2):
                    continue

                pt = line1.intersection(line2)
                if pt.geom_type != "Point":
                    continue

                pt_coords = np.array(pt.coords[0])
                gap_size = 6

                def draw_gapped_segment(a, b):
                    vec = b - a
                    length = np.linalg.norm(vec)
                    if length < 1e-3:
                        return
                    dir = vec / length
                    offset = dir * gap_size
                    self.canvas.create_line(*a, *(pt_coords - offset), fill="black", width=2)
                    self.canvas.create_line(*(pt_coords + offset), *b, fill="black", width=2)

                # Draw gap on underpass
                if seg1.is_overpass:
                    draw_gapped_segment(p2a, p2b)
                    already_drawn.add(seg2.id)
                else:
                    draw_gapped_segment(p1a, p1b)
                    already_drawn.add(seg1.id)

        # Now draw all segments fully (including overpasses)
        for seg in self.segments:
            if seg.id in already_drawn:
                continue
            p1 = self.points[seg.p1].pos
            p2 = self.points[seg.p2].pos
            self.canvas.create_line(*p1, *p2, fill="black", width=2)

        # === Draw points ===
        for pt in self.points:
            x, y = pt.pos
            pid = pt.id

            if pid in self.locked_indices:
                if pt.is_agent:
                    self.canvas.create_oval(x - 6, y - 6, x + 6, y + 6, fill="blue4")
                else:
                    self.canvas.create_oval(x - 6, y - 6, x + 6, y + 6, fill="cornflowerblue")
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

        scale = 40
        canvas_width = int(self.canvas['width'])
        canvas_height = int(self.canvas['height'])

        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)

        knot_width = (max_x - min_x + 1) * scale
        knot_height = (max_y - min_y + 1) * scale

        ox = (canvas_width - knot_width) // 2 - min_x * scale
        oy = (canvas_height - knot_height) // 2 - min_y * scale

        id_map = {}

        # Create KnotPoints
        for p in path_order:
            x, y = p[1] * scale + ox, p[0] * scale + oy
            idx = len(self.points)
            id_map[p] = idx
            self.points.append(KnotPoint(idx, x, y, p in agent_points_set))

        # Create KnotSegments
        for sec in section_list:
            p1, p2 = id_map[sec.start], id_map[sec.end]
            self.segments.append(KnotSegment(len(self.segments), p1, p2, sec.over_under == 1))

        self.ordered_indices = [id_map[p] for p in path_order if p in agent_points_set]

        # Lock first segment's agents and intermediates
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

        # Compute segment crossings
        self.initial_crossings = compute_crossings_from_points(self.points, self.segments)

        # Insert gap info for underpasses
        for (seg1_id, seg2_id), pt in self.initial_crossings.items():
            seg1 = next(s for s in self.segments if s.id == seg1_id)
            seg2 = next(s for s in self.segments if s.id == seg2_id)
            under_seg = seg1 if not seg1.is_overpass else seg2
            if pt.geom_type == "Point":
                under_seg.gap_at.append(pt)

        # Initialize simulation state
        self.velocities = [np.array([0.0, 0.0]) for _ in self.points]
        self.prev_force_dirs = [np.zeros(2, dtype=float) for _ in self.points]
        self.vibrate_count = [0 for _ in self.points]

        # Store equilibrium distances
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
