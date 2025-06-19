import tkinter as tk
import numpy as np

root = tk.Tk()
root.title("Hitch Layers Viewer")
canvas = tk.Canvas(root, width=600, height=600, bg='white')
canvas.pack()

# === Anchors ===
coords = {
    "r1": np.array([280.0, 200.0]),
    "r2": np.array([120.0, 280.0]),
    "r3": np.array([160.0, 40.0]),
    "q1": np.array([240.0, 40.0]),
    "q2": np.array([280.0, 280.0]),
    "q3": np.array([40.0, 200.0]),
}

# === Layer 1
layer1_points = {
    "p1,1": np.array([200.0, 120.0]),
    "p2,1": np.array([240.0, 200.0]),
    "p3,1": np.array([160.0, 200.0]),
}
layer1_cables = [
    ["r3", "p1,1", "p3,1", "q3"],
    ["r1", "p2,1", "p1,1", "q1"],
    ["r2", "p3,1", "p2,1", "q2"],
]

# === Layer 2 (offset from Layer 1)
def offset_midpoint(pa, pb, offset=40):
    mid = 0.5 * (pa + pb)
    edge = pb - pa
    perp = np.array([-edge[1], edge[0]])
    perp = perp / np.linalg.norm(perp)
    return mid + offset * perp

p1_2 = offset_midpoint(layer1_points["p2,1"], layer1_points["p1,1"])
p2_2 = offset_midpoint(layer1_points["p3,1"], layer1_points["p2,1"])
p3_2 = offset_midpoint(layer1_points["p1,1"], layer1_points["p3,1"])

layer2_points = {
    **layer1_points,
    "p1,2": p1_2,
    "p2,2": p2_2,
    "p3,2": p3_2,
}
layer2_cables = [
    ["r3", "p3,2", "p1,1", "p3,1", "p3,2", "q3"],
    ["r1", "p1,2", "p2,1", "p1,1", "p1,2", "q1"],
    ["r2", "p2,2", "p3,1", "p2,1", "p2,2", "q2"],
]

# === Layer 3 (positions manually chosen to match the diagram)
layer3_points = {
    **layer2_points,
    "p1,3": np.array([320.0, 120.0]),
    "p2,3": np.array([320.0, 260.0]),
    "p3,3": np.array([80.0, 140.0]),
}
layer3_cables = [
    ["r3", "p1,3", "p3,2", "p1,1", "p3,1", "p3,2", "p3,3", "q3"],
    ["r1", "p2,3", "p1,2", "p2,1", "p1,1", "p1,2", "p1,3", "q1"],
    ["r2", "p3,3", "p2,2", "p3,1", "p2,1", "p2,2", "p2,3", "q2"],
]

# === Shared State
p_coords = {}
p_velocities = {}
cables = []
colors = ["red", "gold", "blue"]
point_handles = {}
text_labels = {}
cable_handles = []

dragging = False
selected_anchor = None
simulation_running = False

# === Physics
def unit_vector(p1, p2):
    v = p2 - p1
    norm = np.linalg.norm(v)
    return v / norm if norm > 1e-5 else np.zeros_like(v)

def compute_forces():
    forces = {k: np.zeros(2) for k in p_coords}
    for cable in cables:
        for i in range(1, len(cable) - 1):
            a, b, c = cable[i - 1], cable[i], cable[i + 1]
            if b in p_coords:
                p = p_coords[b]
                va = unit_vector(p, coords[a] if a in coords else p_coords[a])
                vc = unit_vector(p, coords[c] if c in coords else p_coords[c])
                forces[b] += va + vc
    return forces

def step():
    if simulation_running:
        forces = compute_forces()
        for k in p_coords:
            dt = 1.0
            damping = 0.95
            p_velocities[k] += dt * forces[k]
            p_velocities[k] *= damping
            p_coords[k] += dt * p_velocities[k]
    redraw()
    root.after(10, step)

# === Drawing
def redraw():
    for h in cable_handles:
        canvas.delete(h)
    cable_handles.clear()

    if all(k in p_coords for k in ["p1,1", "p2,1", "p3,1"]):
        triangle = canvas.create_polygon(
            *p_coords["p1,1"], *p_coords["p2,1"], *p_coords["p3,1"],
            fill='cyan', outline='black', stipple='gray25')
        cable_handles.append(triangle)

    for cable, color in zip(cables, colors):
        pts = [(coords[p] if p in coords else p_coords[p]) for p in cable]
        for i in range(len(pts) - 1):
            line = canvas.create_line(*pts[i], *pts[i + 1], fill=color, width=2)
            cable_handles.append(line)

    for label, pos in {**coords, **p_coords}.items():
        x, y = pos
        if label not in point_handles:
            point_handles[label] = canvas.create_oval(0, 0, 0, 0, fill='black')
        canvas.coords(point_handles[label], x - 4, y - 4, x + 4, y + 4)

        if label not in text_labels:
            text_labels[label] = canvas.create_text(x + 10, y, text=label, anchor=tk.W)
        else:
            canvas.coords(text_labels[label], x + 10, y)

# === Mouse Interaction
def on_press(event):
    global dragging, selected_anchor
    for label, oval in point_handles.items():
        if label in coords:
            x0, y0, x1, y1 = canvas.coords(oval)
            if x0 <= event.x <= x1 and y0 <= event.y <= y1:
                dragging = True
                selected_anchor = label
                break

def on_drag(event):
    if dragging and selected_anchor:
        coords[selected_anchor] = np.array([event.x, event.y])
        redraw()

def on_release(event):
    global dragging, selected_anchor
    dragging = False
    selected_anchor = None

canvas.bind("<ButtonPress-1>", on_press)
canvas.bind("<B1-Motion>", on_drag)
canvas.bind("<ButtonRelease-1>", on_release)

# === Layer Switching
def set_layer(points, cable_list):
    global p_coords, cables, p_velocities, simulation_running
    simulation_running = False
    p_coords.clear()
    p_coords.update(points)
    cables.clear()
    cables.extend(cable_list)
    p_velocities.clear()
    for k in p_coords:
        p_velocities[k] = np.zeros(2)
    redraw()

def start_simulation():
    global simulation_running
    simulation_running = True

# === Buttons
frame = tk.Frame(root)
frame.pack()
tk.Button(frame, text="Layer 1", command=lambda: set_layer(layer1_points, layer1_cables)).pack(side=tk.LEFT)
tk.Button(frame, text="Layer 2", command=lambda: set_layer(layer2_points, layer2_cables)).pack(side=tk.LEFT)
tk.Button(frame, text="Layer 3", command=lambda: set_layer(layer3_points, layer3_cables)).pack(side=tk.LEFT)
tk.Button(frame, text="Start", command=start_simulation).pack(side=tk.LEFT)

# === Init
set_layer(layer1_points, layer1_cables)
step()
root.mainloop()
