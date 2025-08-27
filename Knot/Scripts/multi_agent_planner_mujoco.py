# multi_agent_planner_mujoco.py
# MuJoCo multi-agent path follower with:
# - Connected path lines (capsules)
# - Viewer stays open; loads paused
# - Control-tab "buttons": ui_start (toggle), ui_reload, ui_quit
# Drones are square boxes moved by setting free-joint qpos.

import csv, time, argparse
from tkinter import filedialog, Tk
from math import hypot, sin, pi
from textwrap import dedent

import mujoco as mj
from mujoco import viewer

# ── Config ─────────────────────────────────────────────────────────
PIXEL_TO_METER = 0.01
SCALE_FACTOR   = 2.0
SPHERE_HEIGHT  = 0.05
PATH_RADIUS    = 0.008

DRONE_SPEED    = 0.5
STEP_INTERVAL  = 0.05
DRONE_ALTITUDE = 1.0

COLOR_AGENT     = "1 0 0 1"
COLOR_OVER      = "0.2 0.4 1 1"
COLOR_UNDER     = "0 0.8 0 1"
COLOR_PATH      = "0.2 0.6 1 0.8"
COLOR_PATH_DOT  = "0.2 0.6 1 0.4"

# ── CSV helpers ────────────────────────────────────────────────────
def select_csv_file_dialog():
    root = Tk(); root.withdraw()
    fp = filedialog.askopenfilename(title="Select Path CSV File", filetypes=[("CSV files", "*.csv")])
    if not fp: raise SystemExit("❌ No file selected.")
    return fp

def load_path_from_csv(file_path):
    index_to_point, path_sequence, agent_indices, type_map, entry_point = {}, [], [], {}, None
    with open(file_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            idx   = int(row["Index"])
            x     = float(row["X"]) * PIXEL_TO_METER
            y     = float(row["Y"]) * PIXEL_TO_METER
            label = row.get("Type", "").strip().lower()
            cross = row.get("CrossType", "").strip().lower()
            index_to_point[idx] = (x, y)
            path_sequence.append(idx)
            type_map[idx] = (label, cross)
            if label == "agent": agent_indices.append(idx)
            if idx == 0: entry_point = (x, y)
    if entry_point is None:
        raise RuntimeError("❌ Entry point (Index 0) not found!")
    return index_to_point, path_sequence, agent_indices, entry_point, type_map

def center_and_scale(index_to_point, entry_point):
    xs = [p[0] for p in index_to_point.values()]
    ys = [p[1] for p in index_to_point.values()]
    mx, my = sum(xs)/len(xs), sum(ys)/len(ys)
    for k in list(index_to_point.keys()):
        x, y = index_to_point[k]
        index_to_point[k] = ((x - mx)*SCALE_FACTOR, (y - my)*SCALE_FACTOR)
    ex, ey = entry_point
    return index_to_point, ((ex - mx)*SCALE_FACTOR, (ey - my)*SCALE_FACTOR)

def compute_entry_line_starts(index_to_point, agent_indices, spacing=0.5):
    if 0 not in index_to_point or 1 not in index_to_point:
        raise RuntimeError("❌ Need Index 0 and Index 1 to form entry vector.")
    x0, y0 = index_to_point[0]; x1, y1 = index_to_point[1]
    dx, dy = x0 - x1, y0 - y1
    L = hypot(dx, dy);
    if L == 0: raise RuntimeError("❌ Index 0 and Index 1 are the same point!")
    dx, dy = dx/L, dy/L
    starts = []
    n = len(agent_indices)
    for i in range(n):
        off = spacing * (n - 1 - i)
        starts.append((x0 + dx*off, y0 + dy*off))
    return starts

# ── MJCF build ─────────────────────────────────────────────────────
def build_mujoco_xml(index_to_point, path_sequence, agent_indices, type_map, drone_starts):
    z = SPHERE_HEIGHT

    # Path lines
    path_capsules_xml = []
    for i in range(len(path_sequence) - 1):
        a, b = path_sequence[i], path_sequence[i+1]
        x1, y1 = index_to_point[a]; x2, y2 = index_to_point[b]
        path_capsules_xml.append(
            f'<geom type="capsule" fromto="{x1} {y1} {z} {x2} {y2} {z}" '
            f'size="{PATH_RADIUS}" rgba="{COLOR_PATH}" contype="0" conaffinity="0"/>'
        )

    # Dots (optional)
    path_spheres_xml = []
    for idx in path_sequence:
        x, y = index_to_point[idx]
        path_spheres_xml.append(
            f'<geom type="sphere" size="0.01" pos="{x} {y} {z}" rgba="{COLOR_PATH_DOT}" contype="0" conaffinity="0"/>'
        )

    # Agent/crossing markers
    marker_xml = []
    for idx, (label, cross) in type_map.items():
        x, y = index_to_point[idx]
        if label == "agent":
            marker_xml.append(f'<geom type="sphere" size="0.03" pos="{x} {y} {z}" rgba="{COLOR_AGENT}" contype="0" conaffinity="0"/>')
        elif label == "crossing" and cross == "crossing-over":
            marker_xml.append(f'<geom type="sphere" size="0.02" pos="{x} {y} {z}" rgba="{COLOR_OVER}" contype="0" conaffinity="0"/>')
        elif label == "crossing" and cross == "crossing-under":
            marker_xml.append(f'<geom type="sphere" size="0.02" pos="{x} {y} {z}" rgba="{COLOR_UNDER}" contype="0" conaffinity="0"/>')

    # Drones
    drones_xml = []
    for i, (sx, sy) in enumerate(drone_starts):
        drones_xml.append(dedent(f"""
            <body name="Agent_{i}" pos="{sx} {sy} {DRONE_ALTITUDE}">
                <freejoint name="Agent_{i}_free"/>
                <geom name="Agent_{i}_box" type="box" size="0.05 0.05 0.02" rgba="0.9 0.9 0.9 1"/>
            </body>
        """))

    # Hidden UI controls body (provides harmless joints for our "button" actuators)
    ui_controls_body = dedent(f"""
        <body name="ui_controls" pos="0 0 -10">
            <!-- tiny invisible mass to avoid singular inertia -->
            <geom type="sphere" size="1e-6" rgba="0 0 0 0" density="1000"/>
            <joint name="ui_j_start"  type="slide" axis="1 0 0" range="0 1"/>
            <joint name="ui_j_reload" type="slide" axis="1 0 0" range="0 1"/>
            <joint name="ui_j_quit"   type="slide" axis="1 0 0" range="0 1"/>
        </body>
    """)

    xml = f"""
<mujoco model="knot_agents">
  <option timestep="{STEP_INTERVAL}" integrator="RK4" gravity="0 0 -9.81"/>
  <visual>
    <quality shadowsize="2048"/>
    <map znear="0.01" zfar="50"/>
  </visual>

  <worldbody>
    <light name="light" pos="2 2 5" dir="0 0 -1"/>
    <geom name="ground" type="plane" size="5 5 0.1" rgba="0.95 0.95 0.95 1"/>

    {"".join(path_capsules_xml)}
    {"".join(path_spheres_xml)}
    {"".join(marker_xml)}
    {"".join(drones_xml)}
    {ui_controls_body}
  </worldbody>

  <actuator>
    <!-- gear='0' ensures these actuators exert no torque; they are UI-only -->
    <motor name="ui_start"  joint="ui_j_start"  ctrllimited="true" ctrlrange="0 1" gear="0"/>
    <motor name="ui_reload" joint="ui_j_reload" ctrllimited="true" ctrlrange="0 1" gear="0"/>
    <motor name="ui_quit"   joint="ui_j_quit"   ctrllimited="true" ctrlrange="0 1" gear="0"/>
  </actuator>
</mujoco>
"""
    return xml

# ── One session (returns 'reload'|'quit'|None) ─────────────────────
def run_session(csv_path):
    index_to_point, path_sequence, agent_indices, entry_point, type_map = load_path_from_csv(csv_path)
    index_to_point, entry_point = center_and_scale(index_to_point, entry_point)
    if not agent_indices:
        raise RuntimeError("❌ No agent indices found in CSV (Type == 'agent').")

    drone_starts = compute_entry_line_starts(index_to_point, agent_indices, spacing=0.5)
    xml = build_mujoco_xml(index_to_point, path_sequence, agent_indices, type_map, drone_starts)
    model = mj.MjModel.from_xml_string(xml)
    data  = mj.MjData(model)

    # Map the "button" actuators
    aid_start  = mj.mj_name2id(model, mj.mjtObj.mjOBJ_ACTUATOR, "ui_start")
    aid_reload = mj.mj_name2id(model, mj.mjtObj.mjOBJ_ACTUATOR, "ui_reload")
    aid_quit   = mj.mj_name2id(model, mj.mjtObj.mjOBJ_ACTUATOR, "ui_quit")

    # Crossings
    crossings = {idx: {"pos": index_to_point[idx], "type": ct}
                 for idx, (lbl, ct) in type_map.items() if lbl == "crossing"}

    # Drone state
    z_base, base_arc_radius, amplitude = DRONE_ALTITUDE, 0.5, 0.3
    drone_states = []
    for i in range(len(agent_indices)):
        stop_idx = agent_indices[i]
        sx, sy = drone_starts[i]
        jname = f"Agent_{i}_free"
        jid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, jname)
        qadr = model.jnt_qposadr[jid]
        data.qpos[qadr:qadr+7] = [sx, sy, z_base, 1, 0, 0, 0]
        drone_states.append({"index": i, "stop_index": stop_idx, "current_index": 0,
                             "reached": False, "position": [sx, sy], "qadr": qadr})

    def set_pose(st, x, y, z):
        qadr = st["qadr"]; data.qpos[qadr:qadr+7] = [x, y, z, 1, 0, 0, 0]

    started = False
    printed_hint = False

    with viewer.launch_passive(model, data) as v:
        # overlays: brief instructions
        v.add_overlay(viewer.GridPos.bottomleft,
                      "Controls (Control tab sliders as buttons):",
                      "ui_start>0.5: Start/Pause | ui_reload>0.5: Reload CSV | ui_quit>0.5: Quit")

        while v.is_running():
            # Read "button" sliders (rising-edge detection)
            def pressed(aid):
                # treat >0.5 as a press; then auto-reset to 0
                if aid >= 0 and data.ctrl[aid] > 0.5:
                    data.ctrl[aid] = 0.0
                    return True
                return False

            if pressed(aid_quit):
                print("👋 Quit requested.")
                return "quit"
            if pressed(aid_reload):
                print("🔄 Reload requested.")
                return "reload"
            if pressed(aid_start):
                started = not started
                print("▶️  Start" if started else "⏸️  Paused")

            if not printed_hint:
                printed_hint = True
                print("\nOpen the viewer's **Control** tab and nudge these sliders above 0.5:\n"
                      "  • ui_start  → Start/Pause\n"
                      "  • ui_reload → Reload CSV\n"
                      "  • ui_quit   → Quit\n")

            # If paused, keep static scene
            if not started:
                mj.mj_forward(model, data)
                time.sleep(0.02)
                continue

            # If all done, auto-pause (viewer remains open)
            if all(s["reached"] for s in drone_states):
                if started:
                    started = False
                    print("🎯 All agents deployed. (Paused; viewer stays open.)")
                mj.mj_forward(model, data)
                time.sleep(0.02)
                continue

            # Update each drone
            for st in drone_states:
                if st["reached"]:
                    continue

                ci = st["current_index"]
                if ci >= len(path_sequence) - 1:
                    st["reached"] = True
                    continue

                a, b = path_sequence[ci], path_sequence[ci+1]
                x1, y1 = index_to_point[a]; x2, y2 = index_to_point[b]
                cx, cy = st["position"]
                dx, dy = (x2 - x1), (y2 - y1)
                seg = hypot(dx, dy)
                if seg == 0:
                    st["current_index"] += 1; continue
                dirx, diry = dx/seg, dy/seg

                step = min(DRONE_SPEED*STEP_INTERVAL, hypot(x2 - cx, y2 - cy))
                cx += dirx*step; cy += diry*step
                st["position"] = [cx, cy]

                # altitude arc near crossings (endpoints of segment)
                z_arc = z_base
                cand = []
                if a in crossings: cand.append(("A", crossings[a]))
                if b in crossings: cand.append(("B", crossings[b]))
                if cand:
                    best = None; best_abs = None
                    for where, meta in cand:
                        bx, by = meta["pos"]
                        dfx, dfy = (dirx, diry) if where == "A" else (-dirx, -diry)
                        s = (cx - bx)*dfx + (cy - by)*dfy
                        R = min(base_arc_radius, 0.45*seg)
                        if -R <= s <= R:
                            u = max(-1.0, min(1.0, s/R))
                            off = amplitude * sin((u + 1.0) * pi * 0.5)
                            if best is None or abs(s) < best_abs:
                                best, best_abs = (meta["type"], off), abs(s)
                    if best:
                        ctype, off = best
                        z_arc = z_base + off if ctype == "crossing-over" else z_base - off

                set_pose(st, cx, cy, z_arc)

                # stop at agent index
                sx, sy = index_to_point[st["stop_index"]]
                if hypot(cx - sx, cy - sy) < 0.02:
                    set_pose(st, sx, sy, z_base)
                    st["reached"] = True
                    print(f"📍 Dropped Agent_{st['index']} at Index {st['stop_index']}")

                if hypot(cx - x2, cy - y2) < 1e-4:
                    st["current_index"] += 1

            mj.mj_step(model, data)

    # Viewer closed by user
    return None

# ── Main loop with reload ──────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="MuJoCo multi-agent with Control-tab UI 'buttons'.")
    ap.add_argument("--csv", type=str, default=None, help="Path to CSV (Index,X,Y,Type,CrossType)")
    args = ap.parse_args()

    csv_path = args.csv if args.csv else select_csv_file_dialog()

    while True:
        result = run_session(csv_path)
        if result == "reload":
            csv_path = select_csv_file_dialog()
            continue
        elif result == "quit":
            break
        else:
            # viewer window closed by user
            break

if __name__ == "__main__":
    main()
