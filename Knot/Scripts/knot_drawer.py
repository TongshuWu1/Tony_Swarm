import tkinter as tk
from shapely.geometry import LineString
import numpy as np
from geometry_utils import check_preserve_crossings_and_update_gaps
from region_detection import compute_agent_reduction, knot_manager, read_path
from tkinter import filedialog
import csv


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

        self.physics_running = False

        self.points, self.segments = [], []
        self.initial_crossings = {}
        self.dragging_point = None
        self.original_positions = []
        self.velocities = []
        self.manual_locked_indices = set()
        self.locked_indices = set()
        self.frozen_intermediates = set()
        self.equilibrium_distances = {}

        self.convergence_skip_frames = 5
        self.frames_since_segment_start = 10

        self.converged_counter = 0
        self.convergence_threshold = 0.4
        self.converged_steps_required = 10

        self.avg_speed_buffer = []
        self.avg_speed_window_size = 7

        self.obstacles = []
        self.selected_obstacle_idx = None

        self.prev_force_dirs = [
            np.zeros(2, dtype=float) for _ in range(len(self.points))
        ]
        self.vibrate_count = [0 for _ in range(len(self.points))]

        self.radius = 50
        self.straighten_step = 0
        self.ordered_indices = []

        self.canvas.bind("<ButtonPress-1>", self.on_drag_start)
        self.canvas.bind("<B1-Motion>", self.on_drag_motion)
        self.canvas.bind("<ButtonRelease-1>", self.on_drag_end)

        # Compact param container
        param_container = tk.LabelFrame(
            parent, text="Physics Parameters", padx=5, pady=5
        )
        param_container.pack(padx=10, pady=5)

        param_frame = tk.Frame(param_container)
        param_frame.pack()

        param_labels = [
            ("Spring (k)", "k_entry", 0.04),
            ("Damping (c)", "c_entry", 0.04),
            ("Mass (m)", "m_entry", 0.4),
            ("Time Step (dt)", "dt_entry", 0.4),
            ("Straighten", "straighten_force_entry", 1.5),
            ("Repel Strength", "repulsion_entry", 4.0),
            ("Min Dist", "min_dist_entry", 30.0),
            ("Locked Mult", "locked_repel_multiplier_entry", 10.0),
            ("Conv Thresh", "conv_thresh_entry", 0.0001),
            ("Conv Frames", "conv_steps_entry", 12),
        ]

        for idx, (label, attr, default) in enumerate(param_labels):
            row, col = divmod(idx, 2)
            tk.Label(param_frame, text=label).grid(row=row, column=col * 2, sticky="w")
            entry = tk.Entry(param_frame, width=6)
            entry.insert(0, str(default))
            entry.grid(row=row, column=col * 2 + 1)
            setattr(self, attr, entry)

        # Obstacle radius slider (horizontal, compact)
        slider_frame = tk.Frame(param_container)
        slider_frame.pack(pady=(4, 0))
        tk.Label(slider_frame, text="Obstacle Radius").pack(side="left")
        self.obstacle_radius_slider = tk.Scale(
            slider_frame,
            from_=10,
            to=200,
            orient="horizontal",
            command=self.update_obstacle_radius,
            length=150,
        )
        self.obstacle_radius_slider.set(50)
        self.obstacle_radius_slider.pack(side="left")

        button_frame = tk.Frame(parent)
        button_frame.pack(pady=4)

        self.set_btn = tk.Button(
            button_frame, text="Set", width=10, command=self.update_physics_constants
        )
        self.set_btn.grid(row=0, column=0, padx=2, pady=2)

        self.toggle_btn = tk.Button(
            button_frame, text="Toggle Waypoints", width=15, command=self.redraw
        )
        self.toggle_btn.grid(row=0, column=1, padx=2, pady=2)

        self.next_btn = tk.Button(
            button_frame, text="Next", width=8, command=self.next_segment
        )
        self.next_btn.grid(row=0, column=2, padx=2, pady=2)

        self.add_obstacle_btn = tk.Button(
            button_frame, text="Add Obstacle", width=15, command=self.add_obstacle
        )
        self.add_obstacle_btn.grid(row=1, column=0, padx=2, pady=2)

        self.auto_converge = tk.BooleanVar(value=True)
        self.auto_converge_check = tk.Checkbutton(
            button_frame, text="Auto-Converge", variable=self.auto_converge
        )
        self.auto_converge_check.grid(row=1, column=1, padx=2, pady=2)

        self.start_btn = tk.Button(
            button_frame, text="Start Physics", width=15, command=self.start_physics
        )
        self.start_btn.grid(row=2, column=0, padx=2, pady=2)

        self.save_btn = tk.Button(
            button_frame, text="Save Path", width=15, command=self.save_path_info
        )
        self.save_btn.grid(row=2, column=1, padx=2, pady=2)

        self.update_physics_constants()

    def lock_points(self, point_ids):
        for pid in point_ids:
            self.manual_locked_indices.discard(pid)
            self.locked_indices.add(pid)
        print(f"🔒 Locked points: {point_ids}")

    def update_all_positions_centered(self):
        if not self.points:
            return

        xs = [pt.pos[0] for pt in self.points]
        ys = [pt.pos[1] for pt in self.points]
        cx, cy = sum(xs) / len(xs), sum(ys) / len(ys)

        for pt in self.points:
            x, y = pt.pos
            pt.pos = (x - cx, y - cy)

        print(f"📍 All point positions updated relative to center ({cx:.2f}, {cy:.2f})")

    def save_path_info(self):
        if not self.points:
            print("⚠️ No points to save.")
            return

        filepath = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv")],
            title="Save Full Path with Crossings",
        )
        if not filepath:
            return

        def find_segment(p1_id, p2_id):
            for s in self.segments:
                if (s.p1 == p1_id and s.p2 == p2_id) or (
                    s.p1 == p2_id and s.p2 == p1_id
                ):
                    return s
            return None

        try:
            with open(filepath, "w", newline="") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(["Index", "X", "Y", "Type", "CrossType"])

                written = set()
                next_index = max(pt.id for pt in self.points) + 1

                # Get full path order (agents and turns)
                full_path_ids = []
                for i in range(len(self.ordered_indices) - 1):
                    id_a = self.ordered_indices[i]
                    id_b = self.ordered_indices[i + 1]
                    path_ids = [p.id for p in self.points]
                    idx_a = path_ids.index(id_a)
                    idx_b = path_ids.index(id_b)
                    if idx_a > idx_b:
                        idx_a, idx_b = idx_b, idx_a
                    full_path_ids.extend(path_ids[idx_a:idx_b])
                full_path_ids.append(self.ordered_indices[-1])
                full_path_ids = list(
                    dict.fromkeys(full_path_ids)
                )  # remove duplicates, preserve order

                for i in range(len(full_path_ids) - 1):
                    id_a = full_path_ids[i]
                    id_b = full_path_ids[i + 1]
                    pt_a = self.points[id_a]
                    pt_b = self.points[id_b]

                    seg = find_segment(id_a, id_b)
                    if seg is None:
                        print(f"⚠️ Segment not found between {id_a} and {id_b}")
                        continue

                    # Write point A
                    if id_a not in written:
                        x, y = pt_a.pos
                        point_type = "Agent" if pt_a.is_agent else "Turn"
                        writer.writerow(
                            [pt_a.id, f"{x:.2f}", f"{y:.2f}", point_type, "Straight"]
                        )
                        written.add(id_a)

                    # Insert crossings with prior segments
                    curr_line = LineString([pt_a.pos, pt_b.pos])
                    for other in self.segments:
                        if other.id >= seg.id:
                            continue  # only earlier segments
                        other_line = LineString([self.points[other.p1].pos, self.points[other.p2].pos])
                        if curr_line.crosses(other_line):
                            pt = curr_line.intersection(other_line)
                            if pt.geom_type == "Point":
                                x, y = pt.coords[0]
                                # Patch start: check if this is the last segment
                                is_last_segment = (i == len(full_path_ids) - 2)
                                cross_type = "Crossing-Over" if (
                                            seg.is_overpass or is_last_segment) else "Crossing-Under"
                                # Patch end
                                writer.writerow(
                                    [
                                        next_index,
                                        f"{x:.2f}",
                                        f"{y:.2f}",
                                        "Crossing",
                                        cross_type,
                                    ]
                                )
                                next_index += 1

                    # Write point B
                    if id_b not in written:
                        x, y = pt_b.pos
                        point_type = "Agent" if pt_b.is_agent else "Turn"
                        writer.writerow(
                            [pt_b.id, f"{x:.2f}", f"{y:.2f}", point_type, "Straight"]
                        )
                        written.add(id_b)

            print(f"✅ Full path (all points + crossings) saved to {filepath}")
        except Exception as e:
            print(f"❌ Failed to save file: {e}")

    def update_physics_constants(self):
        try:
            self.k = float(self.k_entry.get())
            self.c = float(self.c_entry.get())
            self.m = float(self.m_entry.get())
            self.dt = float(self.dt_entry.get())
            self.straighten_force = float(self.straighten_force_entry.get())
            self.repulsion_strength = float(self.repulsion_entry.get())
            self.min_dist_threshold = float(self.min_dist_entry.get())
            self.locked_repel_multiplier = float(
                self.locked_repel_multiplier_entry.get()
            )
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
        # Check if clicked near an obstacle center
        for idx, (cx, cy) in enumerate([obs[0] for obs in self.obstacles]):
            if abs(cx - e.x) < 8 and abs(cy - e.y) < 8:
                self.selected_obstacle_idx = idx
                # Sync slider with selected obstacle radius
                self.obstacle_radius_slider.set(self.obstacles[idx][1])
                return

        # Check if clicked on a movable (non-locked) point
        for pt in self.points:
            if pt.id in self.locked_indices:
                continue
            x, y = pt.pos
            if abs(x - e.x) <= 6 and abs(y - e.y) <= 6:
                self.dragging_point = pt
                self.original_positions = [p.pos for p in self.points]
                return

        # If click missed both obstacle and point, clear selection
        self.selected_obstacle_idx = None

    def on_drag_motion(self, e):
        # Dragging an obstacle
        if self.selected_obstacle_idx is not None:
            _, r = self.obstacles[self.selected_obstacle_idx]
            self.obstacles[self.selected_obstacle_idx] = ((e.x, e.y), r)
            self.redraw()
            return

        # Dragging a point
        if self.dragging_point:
            self.points[self.dragging_point.id].pos = (e.x, e.y)
            self.redraw()

    def on_drag_end(self, e):
        # Finish obstacle dragging
        if self.selected_obstacle_idx is not None:
            self.selected_obstacle_idx = None
            return

        # Finish point dragging with crossing check
        if not self.dragging_point:
            return

        idx = self.dragging_point.id
        old_pos = self.original_positions[idx]
        self.points[idx].pos = (e.x, e.y)

        ok, msg = check_crossing_structure_equivalence(
            self.points, self.segments, self.initial_crossings
        )
        if not ok:
            print(f"❌ Reverting drag of point {idx}: {msg}")
            self.points[idx].pos = old_pos
        else:
            print(f"✅ Drag complete for point {idx}")
        self.dragging_point = None
        self.redraw()

    def add_obstacle(self):
        x, y = int(self.canvas["width"]) // 2, int(self.canvas["height"]) // 2
        r = self.obstacle_radius_slider.get()
        self.obstacles.append(((x, y), r))
        self.selected_obstacle_idx = len(self.obstacles) - 1
        self.redraw()

    def update_obstacle_radius(self, value):
        if self.selected_obstacle_idx is not None:
            center, _ = self.obstacles[self.selected_obstacle_idx]
            self.obstacles[self.selected_obstacle_idx] = (center, float(value))
            self.redraw()

    def start_physics(self):
        if self.physics_running:
            return
        self.physics_running = True
        self.start_btn.config(state="disabled")
        print("▶️ Starting physics loop")
        self.run_physics()

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
                active_intermediates = set(path_ids[start_idx + 1 : end_idx])

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
                    if (
                        i not in self.locked_indices
                        and i not in self.manual_locked_indices
                    ):
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
            # Obstacle repulsion
            for i, pt in enumerate(self.points):
                pos = np.array(pt.pos, dtype=float)
                for center, radius in self.obstacles:
                    disp = pos - np.array(center, dtype=float)
                    dist = np.linalg.norm(disp)
                    if dist < 1e-3:
                        disp = np.random.randn(2) * 0.01
                        dist = 1e-3
                    if dist < radius:
                        repel_dir = normalize(disp)
                        delta = min((radius - dist) / radius, 1.0)
                        strength = 300.0 * (delta**2)
                        if dist < radius * 0.7:
                            strength *= 8
                        elif dist < radius * 0.4:
                            strength *= 20
                        repel_force = repel_dir * strength
                        force_map[i] += repel_force
            # Segment-to-obstacle repulsion
            for seg in self.segments:
                p1 = np.array(self.points[seg.p1].pos, dtype=float)
                p2 = np.array(self.points[seg.p2].pos, dtype=float)
                seg_vec = p2 - p1
                seg_len = np.linalg.norm(seg_vec)
                if seg_len == 0:
                    continue
                direction = seg_vec / seg_len

                num_samples = max(2, int(seg_len / 5))  # sample every ~5px
                for k in range(1, num_samples):
                    t = k / num_samples
                    sample_point = p1 * (1 - t) + p2 * t
                    for center, radius in self.obstacles:
                        disp = sample_point - np.array(center)
                        dist = np.linalg.norm(disp)
                        if dist < radius:
                            repel_dir = disp / (dist + 1e-5)
                            delta = min((radius - dist) / radius, 1.0)
                            strength = 200.0 * (delta**2)
                            force = repel_dir * strength

                            # Distribute force to endpoints
                            force_map[seg.p1] += force * (1 - t)
                            force_map[seg.p2] += force * t

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
                        delta = min(
                            (min_dist_threshold - dist) / min_dist_threshold, 0.5
                        )
                        repel_mag = (
                            repulsion_strength
                            * (
                                locked_repel_multiplier
                                if i in self.locked_indices
                                else 1.0
                            )
                            * (delta**2)
                        )
                        repel_force = repel_dir * repel_mag
                        force_map[i] -= repel_force
                        force_map[seg.p1] += 0.5 * repel_force
                        force_map[seg.p2] += 0.5 * repel_force

            for i in active_intermediates:
                f_now = force_map[i]

                if not self.auto_converge.get():
                    # 🛑 If convergence is off, we never freeze points
                    self.prev_force_dirs[i] = f_now
                    continue

                if np.linalg.norm(f_now) > 5.0:
                    # Check if point is inside an obstacle
                    pos = np.array(self.points[i].pos, dtype=float)
                    inside_obstacle = any(
                        np.linalg.norm(pos - np.array(center)) < radius
                        for center, radius in self.obstacles
                    )
                    if not inside_obstacle:
                        print(f"🧊 Force-freezing point {i} due to jitter spike")
                        self.velocities[i] = np.zeros(2)
                        self.frozen_intermediates.add(i)
                    else:
                        print(
                            f"🛑 Skipping freeze for point {i} — still inside obstacle"
                        )
                    continue

                self.prev_force_dirs[i] = f_now

            for i, pti in enumerate(self.points):
                force = force_map[i]
                force_norm = np.linalg.norm(force)

                if pti == self.dragging_point:
                    continue

                # Allow movement for locked/frozen if force is strong enough (esp. obstacle repulsion)
                is_repel_escape = force_norm > 50.0
                if (
                    i in self.locked_indices or i in self.frozen_intermediates
                ) and not is_repel_escape:
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
                ok, _ = check_crossing_structure_equivalence(
                    self.points, self.segments, self.initial_crossings
                )
                if not ok:
                    pti.pos = old_pos
                    self.velocities[i] = np.zeros(2)

            if self.frames_since_segment_start <= self.convergence_skip_frames:
                pass
            else:
                intermediates = [
                    i
                    for i in self.manual_locked_indices
                    if i not in self.locked_indices
                ]
                if not intermediates:
                    self.next_segment()
                    self.converged_counter = 0
                    self.frames_since_segment_start = 0
                elif all(i in self.frozen_intermediates for i in intermediates):
                    stuck_inside = False
                    for i in intermediates:
                        pt_pos = np.array(self.points[i].pos, dtype=float)
                        for center, radius in self.obstacles:
                            if np.linalg.norm(pt_pos - np.array(center)) < radius:
                                print(f"⛔ Frozen point {i} still inside obstacle")
                                stuck_inside = True
                                break
                        if stuck_inside:
                            break

                    if not stuck_inside:
                        print("✅ All intermediates frozen and clear of obstacles.")
                        if self.auto_converge.get():
                            print("➡️ Auto-converge: advancing to next segment.")
                            self.next_segment()
                            self.converged_counter = 0
                            self.frames_since_segment_start = 0
                        else:
                            print(
                                "⏸️ Auto-converge disabled — waiting for manual advance."
                            )
                    else:
                        print("🛑 Obstacle violation — holding segment.")
                else:
                    moving_speeds = [
                        np.linalg.norm(self.velocities[i])
                        for i in intermediates
                        if i not in self.frozen_intermediates
                    ]
                    if not hasattr(self, "avg_speed_buffer"):
                        self.avg_speed_buffer = []
                        self.avg_speed_window_size = 10
                    instant_speed = np.mean(moving_speeds) if moving_speeds else 0.0
                    if instant_speed > 1.0:
                        print(
                            f"⚠️ Detected speed spike: {instant_speed:.4f} — purging buffer"
                        )
                        self.avg_speed_buffer.clear()
                        self.converged_counter = 0
                    else:
                        self.avg_speed_buffer.append(instant_speed)
                        if len(self.avg_speed_buffer) > self.avg_speed_window_size:
                            self.avg_speed_buffer.pop(0)
                        avg_speed = np.mean(self.avg_speed_buffer)
                        print(
                            f"🔍 Instant speed = {instant_speed:.5f} | Smoothed avg = {avg_speed:.5f} | Stable frames: {self.converged_counter}/{self.converged_steps_required}"
                        )
                        if avg_speed < self.convergence_threshold:
                            self.converged_counter += 1
                        else:
                            self.converged_counter = 0
                        if self.converged_counter >= self.converged_steps_required:
                            print("✅ Speed convergence reached.")
                            if self.auto_converge.get():
                                print("➡️ Advancing to next segment.")
                                self.next_segment()
                                self.converged_counter = 0
                                self.frames_since_segment_start = 0
                            else:
                                print(
                                    "⏸️ Auto-converge disabled — waiting for manual advance."
                                )

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
            curr = {
                self.ordered_indices[self.straighten_step],
                self.ordered_indices[self.straighten_step + 1],
            }
        elif self.straighten_step < len(self.ordered_indices):
            curr = {self.ordered_indices[self.straighten_step]}

        # === Draw segments with over/under gap handling ===
        drawn_pairs = set()
        already_drawn = set()

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
                    self.canvas.create_line(
                        *a, *(pt_coords - offset), fill="black", width=2
                    )
                    self.canvas.create_line(
                        *(pt_coords + offset), *b, fill="black", width=2
                    )

                if seg1.is_overpass:
                    draw_gapped_segment(p2a, p2b)
                    already_drawn.add(seg2.id)
                else:
                    draw_gapped_segment(p1a, p1b)
                    already_drawn.add(seg1.id)

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
                    self.canvas.create_oval(
                        x - 6, y - 6, x + 6, y + 6, fill="cornflowerblue"
                    )
            elif pid in self.manual_locked_indices:
                self.canvas.create_oval(x - 6, y - 6, x + 6, y + 6, fill="skyblue")
            elif pid in curr and pid not in self.locked_indices:
                self.canvas.create_oval(x - 6, y - 6, x + 6, y + 6, fill="gold")
            elif pt.is_agent:
                self.canvas.create_oval(
                    x - 6, y - 6, x + 6, y + 6, fill="white", outline="red", width=2
                )
            else:
                self.canvas.create_oval(x - 4, y - 4, x + 4, y + 4, fill="black")

        # === Draw obstacles ===
        for center, radius in self.obstacles:
            cx, cy = center
            self.canvas.create_oval(
                cx - radius,
                cy - radius,
                cx + radius,
                cy + radius,
                outline="red",
                width=2,
            )
            self.canvas.create_oval(cx - 4, cy - 4, cx + 4, cy + 4, fill="red")

    def draw_sections(self, section_list, agent_points_set):
        self.clear()
        path_order = [section_list[0].start]
        for sec in section_list:
            if (
                sec.end != path_order[-1]
            ):  # avoid duplicate if last end matches previous
                path_order.append(sec.end)

        xs = [p[1] for p in path_order]
        ys = [p[0] for p in path_order]

        scale = 40
        canvas_width = int(self.canvas["width"])
        canvas_height = int(self.canvas["height"])

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

        # Create KnotSegments, skipping duplicates
        existing_seg_pairs = set()
        for sec in section_list:
            p1, p2 = id_map[sec.start], id_map[sec.end]
            pair = tuple(sorted((p1, p2)))  # order-independent key
            if pair in existing_seg_pairs:
                print(
                    f"⚠️ Skipping duplicate segment between {self.points[p1].pos} and {self.points[p2].pos}"
                )
                continue
            existing_seg_pairs.add(pair)
            self.segments.append(
                KnotSegment(len(self.segments), p1, p2, sec.over_under == 1)
            )

        # 🔍 Print segment info
        print("📏 Segment List:")
        for seg in self.segments:
            start_pt = self.points[seg.p1].pos
            end_pt = self.points[seg.p2].pos
            print(
                f"  Segment {seg.id}: Start {start_pt}, End {end_pt}, Overpass: {seg.is_overpass}"
            )

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
        self.initial_crossings = compute_crossings_from_points(
            self.points, self.segments
        )

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
        app.full_path_list = (
            _  # ⬅️ This stores the traced path from compute_agent_reduction
        )
        agents = {p.pos_2d() for p in knot_manager.agent_registry.values()}
        app.draw_sections(secs, agents)

    except Exception as e:
        print("⚠️", e)
    root.mainloop()
