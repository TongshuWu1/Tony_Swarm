import csv
import tkinter as tk
from tkinter import filedialog
from math import hypot, sin, pi, cos
import time
from coppeliasim_zmqremoteapi_client import RemoteAPIClient

# ───── Configuration ─────
PIXEL_TO_METER = 0.01
SCALE_FACTOR = 2.0
SPHERE_HEIGHT = 0.05

# For windows path
DRONE_MODEL_PATH = "models/robots/mobile/Quadcopter.ttm"

# For macos path
# DRONE_MODEL_PATH = "/Applications/coppeliaSim.app/Contents/Resources/models/robots/mobile/Quadcopter.ttm"


DRONE_SPEED = 0.5        # meters per second
STEP_INTERVAL = 0.05     # seconds per simulation step
DRONE_ALTITUDE = 1.0

def select_csv_file():
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        title="Select Path CSV File",
        filetypes=[("CSV files", "*.csv")]
    )
    if not file_path:
        print("❌ No file selected.")
        exit()
    return file_path

def load_path_from_csv(file_path):
    index_to_point = {}
    path_sequence = []
    agent_indices = []
    type_map = {}
    entry_point = None

    with open(file_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            index = int(row["Index"])
            x = float(row["X"]) * PIXEL_TO_METER
            y = float(row["Y"]) * PIXEL_TO_METER
            label = row.get("Type", "").strip().lower()
            cross = row.get("CrossType", "").strip().lower()

            index_to_point[index] = (x, y)
            path_sequence.append(index)
            type_map[index] = (label, cross)

            if label == "agent":
                agent_indices.append(index)
            if index == 0:
                entry_point = (x, y)

    if entry_point is None:
        raise RuntimeError("❌ Entry point (Index 0) not found!")

    return index_to_point, path_sequence, agent_indices, entry_point, type_map

def center_and_scale(index_to_point, entry_point):
    xs = [pt[0] for pt in index_to_point.values()]
    ys = [pt[1] for pt in index_to_point.values()]
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)

    for index in index_to_point:
        x, y = index_to_point[index]
        x = (x - mean_x) * SCALE_FACTOR
        y = (y - mean_y) * SCALE_FACTOR
        index_to_point[index] = (x, y)

    entry_x, entry_y = entry_point
    entry_point = ((entry_x - mean_x) * SCALE_FACTOR, (entry_y - mean_y) * SCALE_FACTOR)
    return index_to_point, entry_point

def cleanup_scene(sim):
    all_objs = sim.getObjectsInTree(sim.handle_scene, sim.handle_all, 0)
    for obj in all_objs:
        alias = sim.getObjectAlias(obj)
        if alias.startswith("Agent_") or alias.startswith("Point_") or alias == "KnotRawPath":
            sim.removeObject(obj)

def draw_path(sim, index_to_point, path_sequence):
    line_handle = sim.addDrawingObject(
        sim.drawing_lines,
        3,
        0.0,
        -1,
        len(path_sequence) * 2,
        [0.2, 0.6, 1.0]
    )

    for i in range(len(path_sequence) - 1):
        a = path_sequence[i]
        b = path_sequence[i + 1]
        x1, y1 = index_to_point[a]
        x2, y2 = index_to_point[b]
        sim.addDrawingObjectItem(line_handle, [x1, y1, SPHERE_HEIGHT, x2, y2, SPHERE_HEIGHT])

def draw_path_points(sim, index_to_point, type_map):
    for index, (x, y) in index_to_point.items():
        label, cross_type = type_map.get(index, ("", ""))
        z = SPHERE_HEIGHT

        if label == "turn":
            continue

        if label == "agent":
            color = [1.0, 0.0, 0.0]  # red
            size = 0.05
        elif label == "crossing" and cross_type == "crossing-over":
            color = [0.2, 0.4, 1.0]  # blue
            size = 0.035
        elif label == "crossing" and cross_type == "crossing-under":
            color = [0.0, 0.8, 0.0]  # green
            size = 0.035
        else:
            continue

        sphere = sim.createPrimitiveShape(1, [size, size, size])
        sim.setObjectPosition(sphere, -1, [x, y, z])
        sim.setShapeColor(sphere, None, sim.colorcomponent_ambient_diffuse, color)
        sim.setObjectAlias(sphere, f"Point_{index}")

def scale_drone(sim, drone_handle, scale_factor=1):
    shapes = sim.getObjectsInTree(drone_handle, sim.object_shape_type, 0)
    for shape in shapes:
        size = sim.getObjectFloatParam(shape, sim.objfloatparam_objbbox_max_x) - sim.getObjectFloatParam(shape, sim.objfloatparam_objbbox_min_x)
        if size == 0:
            continue
        sim.scaleObject(shape, scale_factor, scale_factor, scale_factor, 0)

def spawn_drones_along_line(sim, index_to_point, agent_indices, entry_point):
    drones = []
    targets = []
    drone_start_positions = []

    x0, y0 = index_to_point[0]
    x1, y1 = index_to_point[1]
    dx = x0 - x1
    dy = y0 - y1
    length = hypot(dx, dy)
    if length == 0:
        raise RuntimeError("❌ Index 0 and Index 1 are the same point!")

    dx /= length
    dy /= length
    spacing = 0.5
    num_agents = len(agent_indices)
    for i in range(num_agents):
        offset = spacing * (num_agents - 1 - i)
        start_x = x0 + dx * offset
        start_y = y0 + dy * offset
        drone_start_positions.append((start_x, start_y))

        drone = sim.loadModel(DRONE_MODEL_PATH)
        sim.setObjectPosition(drone, -1, [start_x, start_y, DRONE_ALTITUDE])
        sim.setObjectAlias(drone, f"Agent_{i}")

        for shape in sim.getObjectsInTree(drone, sim.object_shape_type):
            sim.setObjectInt32Param(shape, sim.shapeintparam_respondable, 0)

        scale_drone(sim, drone)

        all_objects = sim.getObjectsInTree(drone, sim.handle_all, 0)
        target = next((obj for obj in all_objects if "target" in sim.getObjectAlias(obj).lower()), None)
        if target is None:
            raise RuntimeError(f"❌ Could not find target dummy in Agent_{i}")

        drones.append(drone)
        targets.append(target)

    print(f"🚁 Spawned {len(drones)} drones along the entry vector.")
    return drones, targets, drone_start_positions
def move_targets_along_path(sim, drones, targets, index_to_point, path_sequence, agent_indices, drone_start_positions):
    z_base = DRONE_ALTITUDE
    base_arc_radius = 0.5   # desired half-width of the arc along the segment
    amplitude = 0.3         # peak climb (over) / dive (under) at the crossing center

    # Map each crossing index -> {pos, type}
    crossings = {}
    for idx, (label, cross_type) in type_map.items():
        if label == "crossing":
            crossings[idx] = {
                "pos": index_to_point[idx],
                "type": cross_type,  # "crossing-over" or "crossing-under"
            }

    num_drones = len(drones)
    drone_states = []

    for i in range(num_drones):
        stop_index = agent_indices[i] if i < len(agent_indices) else path_sequence[-1]
        state = {
            "drone": drones[i],
            "target": targets[i],
            "stop_index": stop_index,
            "reached": False,
            "current_index": 0,          # segment = path_sequence[current_index] -> next
            "position": list(drone_start_positions[i]),
            "index": i
        }
        sim.setObjectPosition(targets[i], -1, [*state["position"], z_base])
        drone_states.append(state)

    print("▶️ Simulation started.")
    done_count = 0

    while done_count < num_drones:
        for state in drone_states:
            if state["reached"]:
                continue

            current_idx = state["current_index"]
            if current_idx >= len(path_sequence) - 1:
                state["reached"] = True
                done_count += 1
                continue

            a = path_sequence[current_idx]
            b = path_sequence[current_idx + 1]
            x1, y1 = index_to_point[a]
            x2, y2 = index_to_point[b]

            cx, cy = state["position"]
            dx = x2 - x1
            dy = y2 - y1
            seg_len = hypot(dx, dy)
            if seg_len == 0:
                state["current_index"] += 1
                continue

            dir_x = dx / seg_len
            dir_y = dy / seg_len

            # Move along the segment
            move_step = DRONE_SPEED * STEP_INTERVAL
            # If we’re very close to the end, clamp this step to avoid overshoot jitter
            to_end = hypot((x2 - cx), (y2 - cy))
            step = min(move_step, to_end)
            cx += dir_x * step
            cy += dir_y * step
            state["position"] = [cx, cy]

            # ───── Altitude calculation: ONLY when this segment touches a crossing ─────
            z_arc = z_base

            # Identify if this segment is adjacent to a crossing index
            candidates = []
            if a in crossings:
                candidates.append(("at_a", a, crossings[a]))
            if b in crossings:
                candidates.append(("at_b", b, crossings[b]))

            if candidates:
                # Use the closest applicable crossing on this segment (usually just one)
                best = None
                best_abs_s = None

                for where, cross_idx, meta in candidates:
                    (bx, by) = meta["pos"]

                    # Direction pointing *away* from the crossing along this segment
                    # so that s<0 is "before", s=0 at crossing, s>0 "after".
                    if where == "at_a":
                        dir_from_cross_x = dir_x
                        dir_from_cross_y = dir_y
                    else:  # where == "at_b"
                        dir_from_cross_x = -dir_x
                        dir_from_cross_y = -dir_y

                    # Signed distance along the segment from the crossing to current pos.
                    s = (cx - bx) * dir_from_cross_x + (cy - by) * dir_from_cross_y

                    # Limit arc half-width so it fits within the segment neatly.
                    arc_radius = min(base_arc_radius, 0.45 * seg_len)

                    if -arc_radius <= s <= arc_radius:
                        # Map s in [-R, R] to a smooth arch: 0 at edges, max at s=0
                        u = max(-1.0, min(1.0, s / arc_radius))
                        phase = (u + 1.0) * pi * 0.5           # [-1,1] -> [0, π]
                        offset = amplitude * sin(phase)        # 0→max→0

                        # Choose nearest crossing on this segment if two exist (rare)
                        if (best is None) or (abs(s) < best_abs_s):
                            best = (meta["type"], offset)
                            best_abs_s = abs(s)

                if best is not None:
                    cross_type, offset = best
                    if cross_type == "crossing-over":
                        z_arc = z_base + offset
                    elif cross_type == "crossing-under":
                        z_arc = z_base - offset

            # Apply new target position
            sim.setObjectPosition(state["target"], -1, [cx, cy, z_arc])

            # Stop this drone at its assigned agent index
            stop_idx = state["stop_index"]
            stop_x, stop_y = index_to_point[stop_idx]
            if hypot(cx - stop_x, cy - stop_y) < 0.02:
                sim.setObjectPosition(state["target"], -1, [stop_x, stop_y, z_base])
                state["reached"] = True
                done_count += 1
                print(f"📍 Dropped Agent_{state['index']} at Index {stop_idx}")

            # Advance to next segment when we reach the end of the current one
            if hypot(cx - x2, cy - y2) < 1e-4:
                state["current_index"] += 1

        time.sleep(STEP_INTERVAL)

    print("🎯 All agents deployed.")



def main():
    csv_path = select_csv_file()
    index_to_point, path_sequence, agent_indices, entry_point, type_map_loaded = load_path_from_csv(csv_path)
    global type_map
    type_map = type_map_loaded

    index_to_point, entry_point = center_and_scale(index_to_point, entry_point)

    client = RemoteAPIClient()
    sim = client.getObject("sim")

    cleanup_scene(sim)
    draw_path(sim, index_to_point, path_sequence)
    draw_path_points(sim, index_to_point, type_map)

    drones, targets, drone_start_positions = spawn_drones_along_line(sim, index_to_point, agent_indices, entry_point)

    print("⏳ Waiting for simulation to start...")
    while sim.getSimulationState() == sim.simulation_stopped:
        time.sleep(0.1)

    print("▶️ Simulation started.")
    move_targets_along_path(sim, drones, targets, index_to_point, path_sequence, agent_indices, drone_start_positions)
    print("🎯 All agents deployed.")

if __name__ == "__main__":
    main()
