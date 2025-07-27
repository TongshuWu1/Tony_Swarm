import csv
import time
import os
from math import hypot
from coppeliasim_zmqremoteapi_client import RemoteAPIClient
import tkinter as tk
from tkinter import filedialog

# ───── Configuration ─────
SCALE_FACTOR = 2.0          # Multiply size of the path
PIXEL_TO_METER = 0.01       # Convert from pixels to meters
SPHERE_HEIGHT = 0.05        # Height of marker spheres
DRONE_SPEED = 1.0           # Speed in meters per second
DEBUG_LOG = "Knot/path_info/debug_log.csv"

# ───── GUI File Dialog ─────
root = tk.Tk()
root.withdraw()

csv_path = filedialog.askopenfilename(
    title="Select Path CSV File",
    filetypes=[("CSV files", "*.csv")]
)
if not csv_path:
    print("❌ No file selected.")
    exit()

# ───── Connect to CoppeliaSim ─────
client = RemoteAPIClient()
sim = client.getObject("sim")

model_path = "models/robots/mobile/Quadcopter.ttm"
agent_indices = []
index_to_point = {}
entry_point = None
path_sequence = []

# ───── Read and Parse CSV ─────
with open(csv_path, newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        index = int(row["Index"])
        x = float(row["X"]) * PIXEL_TO_METER
        y = float(row["Y"]) * PIXEL_TO_METER
        label = row["Type"].strip().lower()

        index_to_point[index] = (x, y)
        path_sequence.append(index)

        if label == "agent":
            agent_indices.append(index)
        if index == 0:
            entry_point = (x, y)

if entry_point is None:
    raise RuntimeError("❌ Entry point (Index 0) not found!")

# ───── Center the Path on (0, 0) and Apply Scaling ─────
xs = [pt[0] for pt in index_to_point.values()]
ys = [pt[1] for pt in index_to_point.values()]
mean_x = sum(xs) / len(xs)
mean_y = sum(ys) / len(ys)

for index in index_to_point:
    x, y = index_to_point[index]
    x_centered = (x - mean_x) * SCALE_FACTOR
    y_centered = (y - mean_y) * SCALE_FACTOR
    index_to_point[index] = (x_centered, y_centered)

entry_x, entry_y = entry_point
entry_point = ((entry_x - mean_x) * SCALE_FACTOR, (entry_y - mean_y) * SCALE_FACTOR)

# ───── Visualize Path as Connecting Lines ─────
line_handle = sim.addDrawingObject(
    sim.drawing_lines,
    3,
    0.0,
    -1,
    len(path_sequence)*2,
    [0.2, 0.6, 1.0]
)

for i in range(len(path_sequence) - 1):
    a = path_sequence[i]
    b = path_sequence[i + 1]
    x1, y1 = index_to_point[a]
    x2, y2 = index_to_point[b]
    sim.addDrawingObjectItem(line_handle, [x1, y1, SPHERE_HEIGHT, x2, y2, SPHERE_HEIGHT])

# ───── Visualize Agent & Turn Points ─────
for index in path_sequence:
    x, y = index_to_point[index]
    z = SPHERE_HEIGHT

    if index in agent_indices:
        sphere = sim.createPrimitiveShape(1, [0.05, 0.05, 0.05])
        sim.setObjectPosition(sphere, -1, [x, y, z])
        sim.setShapeColor(sphere, None, sim.colorcomponent_ambient_diffuse, [1, 0, 0])
        sim.setObjectAlias(sphere, f"AgentPoint_{index}")
    else:
        sphere = sim.createPrimitiveShape(1, [0.03, 0.03, 0.03])
        sim.setObjectPosition(sphere, -1, [x, y, z])
        sim.setShapeColor(sphere, None, sim.colorcomponent_ambient_diffuse, [0.6, 0.6, 0.6])
        sim.setObjectAlias(sphere, f"TurnPoint_{index}")

# ───── Sort agents by proximity to entry ─────
agent_indices = sorted(agent_indices, key=lambda i: hypot(
    index_to_point[i][0] - entry_point[0],
    index_to_point[i][1] - entry_point[1]
))

# ───── Spawn Drones at Entry Point ─────
spawned_drones = []
num_drones = len(agent_indices)  # one drone per agent point

for i in range(num_drones):
    drone = sim.loadModel(model_path)
    sim.setObjectPosition(drone, -1, [entry_point[0], entry_point[1], 1.0])
    sim.setObjectAlias(drone, f"Agent_{i+1}")

    for shape in sim.getObjectsInTree(drone, sim.object_shape_type):
        sim.setObjectInt32Param(shape, sim.shapeintparam_respondable, 0)
        sim.scaleObject(shape, 0.5, 0.5, 0.5, 0)

    spawned_drones.append(drone)

print(f"✅ {len(spawned_drones)} drones spawned at entry point.")

# ───── Wait for Simulation to Start ─────
print("⏳ Waiting for simulation to start...")
while sim.getSimulationState() == sim.simulation_stopped:
    time.sleep(0.1)
print("▶️ Simulation started.")

# ───── Log Setup ─────
os.makedirs(os.path.dirname(DEBUG_LOG), exist_ok=True)
with open(DEBUG_LOG, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Time", "Agent", "X", "Y", "Z", "Status"])

# ───── Coordinated Agent Movement Along Path ─────
last_agent = spawned_drones[-1]
sim.setObjectPosition(last_agent, -1, [entry_point[0], entry_point[1], 1.0])
print(f"📍 Last agent left at start (Index 0) as Agent_{len(spawned_drones)}")

moving_drones = spawned_drones[:-1]

for i, path_index in enumerate(path_sequence[1:], start=1):  # skip index 0
    x, y = index_to_point[path_index]
    z = 1.0

    for drone in moving_drones:
        current_pos = sim.getObjectPosition(drone, -1)
        dx = x - current_pos[0]
        dy = y - current_pos[1]
        dist = hypot(dx, dy)
        steps = max(int(dist / 0.05), 1)

        for step in range(steps):
            ix = current_pos[0] + dx * (step + 1) / steps
            iy = current_pos[1] + dy * (step + 1) / steps
            sim.setObjectPosition(drone, -1, [ix, iy, z])

            with open(DEBUG_LOG, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([f"{time.time():.2f}", sim.getObjectAlias(drone), f"{ix:.2f}", f"{iy:.2f}", z, "moving"])

            time.sleep(0.05)

    if path_index in agent_indices and path_index != 0 and len(moving_drones) > 0:
        dropped_drone = moving_drones.pop()
        sim.setObjectPosition(dropped_drone, -1, [x, y, z])
        agent_num = spawned_drones.index(dropped_drone) + 1
        print(f"📍 Dropped Agent_{agent_num} at Index {path_index}")

        with open(DEBUG_LOG, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([f"{time.time():.2f}", sim.getObjectAlias(dropped_drone), f"{x:.2f}", f"{y:.2f}", z, "dropped"])

print("🎯 Path complete. All agents placed.")
