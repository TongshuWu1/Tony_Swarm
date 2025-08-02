import csv
import tkinter as tk
from tkinter import filedialog
from math import hypot
import time
from coppeliasim_zmqremoteapi_client import RemoteAPIClient

# ───── Configuration ─────
PIXEL_TO_METER = 0.01
SCALE_FACTOR = 2.0
SPHERE_HEIGHT = 0.05
DRONE_MODEL_PATH = "models/robots/mobile/Quadcopter.ttm"

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

        # === Skip turn points (no visual) ===
        if label == "turn":
            continue

        # === Determine color ===
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
            continue  # unhandled type

        # === Draw sphere ===
        sphere = sim.createPrimitiveShape(1, [size, size, size])
        sim.setObjectPosition(sphere, -1, [x, y, z])
        sim.setShapeColor(sphere, None, sim.colorcomponent_ambient_diffuse, color)
        sim.setObjectAlias(sphere, f"Point_{index}")

def scale_drone(sim, drone_handle, scale_factor=1):
    # Get all shapes inside the drone model
    shapes = sim.getObjectsInTree(drone_handle, sim.object_shape_type, 0)

    for shape in shapes:
        size = sim.getObjectFloatParam(shape, sim.objfloatparam_objbbox_max_x) - sim.getObjectFloatParam(shape, sim.objfloatparam_objbbox_min_x)
        if size == 0:
            continue  # skip empty objects
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

    # Normalize direction vector
    dx /= length
    dy /= length

    spacing =0.5  # Distance between drones
    num_agents = len(agent_indices)
    for i in range(num_agents):
        offset = spacing * (num_agents - 1 - i)  # reversed so Agent_1 is at the back
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
    z = DRONE_ALTITUDE
    num_drones = len(drones)

    agent_to_index = {i: agent_indices[i] for i in range(num_drones)}

    drone_states = []
    for i in range(num_drones):
        state = {
            "drone": drones[i],
            "target": targets[i],
            "stop_index": agent_to_index[i],
            "reached": False,
            "current_index": 0,
            "position": list(drone_start_positions[i]),
        }
        sim.setObjectPosition(drones[i], -1, [state["position"][0], state["position"][1], z])
        sim.setObjectPosition(targets[i], -1, [state["position"][0], state["position"][1], z])
        drone_states.append(state)

    print("▶️ Simulation started.")

    done_count = 0
    while done_count < num_drones:
        for state in drone_states:
            if state["reached"]:
                continue

            current_idx = state["current_index"]
            if current_idx >= len(path_sequence) - 1:
                continue

            a = path_sequence[current_idx]
            b = path_sequence[current_idx + 1]
            x1, y1 = index_to_point[a]
            x2, y2 = index_to_point[b]

            cx, cy = state["position"]
            dx = x2 - x1
            dy = y2 - y1
            segment_length = hypot(dx, dy)

            # Direction of the path segment
            dir_x = dx / segment_length
            dir_y = dy / segment_length

            move_step = DRONE_SPEED * STEP_INTERVAL
            cx += dir_x * move_step
            cy += dir_y * move_step
            state["position"] = [cx, cy]

            # Move target
            state["target_pos"] = [cx, cy, z]
            sim.setObjectPosition(state["target"], -1, state["target_pos"])

            # Move drone to follow target
            sim.setObjectPosition(state["drone"], -1, state["target_pos"])

            # Check if passed the stop index
            stop_idx = state["stop_index"]
            stop_x, stop_y = index_to_point[stop_idx]
            if hypot(cx - stop_x, cy - stop_y) < 0.02:
                sim.setObjectPosition(state["drone"], -1, [stop_x, stop_y, z])
                sim.setObjectPosition(state["target"], -1, [stop_x, stop_y, z])
                state["reached"] = True
                done_count += 1
                print(f"📍 Dropped Agent_{drone_states.index(state)+1} at Index {stop_idx}")

            # Move to next segment if passed midpoint
            if hypot(cx - x2, cy - y2) < move_step:
                state["current_index"] += 1

        time.sleep(STEP_INTERVAL)

    print("🎯 All agents deployed.")


def main():
    csv_path = select_csv_file()
    index_to_point, path_sequence, agent_indices, entry_point, type_map = load_path_from_csv(csv_path)

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