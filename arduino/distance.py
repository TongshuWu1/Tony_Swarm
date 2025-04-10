
import os
import serial
import struct
import time
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.animation import FuncAnimation
from datetime import datetime
from collections import deque
import numpy as np
from scipy.interpolate import make_interp_spline  # NEW

# === Config ===
PORT = '/dev/cu.usbmodem11201'  # Change to your port
BAUD = 115200
ALPHA = 0.3  # Smoothing factor for recorded plot
READ_INTERVAL = 0.1  # seconds between reads
SPIKE_THRESHOLD = 1  # Max difference from median to accept a new value (in meters)
ROLLING_WINDOW_SIZE = 5

# === Serial Setup ===
ser = serial.Serial(PORT, BAUD, timeout=1)
print("Listening for distance data...")

# === Data Storage ===
timestamps = []
distances_m = []
recording = False
recorded_times = []
recorded_distances_m = []
start_time = None
last_read_time = 0
rolling_window = deque(maxlen=ROLLING_WINDOW_SIZE)

# === Plot Setup ===
fig, ax = plt.subplots()
line_dist, = ax.plot([], [], label='Distance (m)', color='tab:blue', linewidth=2.0, alpha=0.9)
line_dist.set_animated(True)

ax.set_xlabel("Time")
ax.set_ylabel("Distance (m)")
ax.set_title("Live Distance from TFmini")
ax.legend(loc='upper left')
ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
plt.xticks(rotation=45)
plt.tight_layout()

# === Live Update Function ===
def update(frame):
    global recording, start_time, last_read_time

    current_time = time.time()
    if current_time - last_read_time < READ_INTERVAL:
        return []

    last_read_time = current_time
    updated = False

    while ser.in_waiting >= 3:
        raw = ser.read(3)
        if raw[0] != ord('D'):
            continue
        distance_cm = struct.unpack('<H', raw[1:])[0]
        distance_m = distance_cm / 100.0

        # === Outlier rejection using rolling median ===
        if len(rolling_window) >= 3:
            median = np.median(rolling_window)
            if abs(distance_m - median) > SPIKE_THRESHOLD:
                print(f"[WARN] Spike filtered: {distance_m:.2f} m (median: {median:.2f})")
                continue

        rolling_window.append(distance_m)

        now = datetime.now()
        timestamps.append(now)
        distances_m.append(distance_m)
        print(f"[{now.strftime('%H:%M:%S')}] Distance: {distance_m:.2f} m")

        if recording:
            duration = time.time() - start_time
            recorded_times.append(duration)
            recorded_distances_m.append(distance_m)

        updated = True

    if updated:
        time_nums = mdates.date2num(timestamps)
        line_dist.set_data(time_nums, distances_m)
        ax.set_xlim(time_nums[0], time_nums[-1])
        ax.set_ylim(min(distances_m) - 0.1, max(distances_m) + 0.1)

    return [line_dist]

# === Keyboard Interaction ===
def on_key(event):
    global recording, start_time, timestamps, distances_m, rolling_window

    if event.key == 'r' and not recording:
        print("[INFO] Started recording...")
        recording = True
        recorded_times.clear()
        recorded_distances_m.clear()
        start_time = time.time()

    elif event.key == 't' and recording:
        print("[INFO] Stopping and saving recording...")
        recording = False

        # Create folder if it doesn't exist
        folder_name = "weight360.4_white"
        os.makedirs(folder_name, exist_ok=True)

        # Generate timestamped filenames
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_filename = os.path.join(folder_name, f"recorded_distance_{timestamp}.csv")
        plot_filename = os.path.join(folder_name, f"recorded_distance_plot_{timestamp}.png")

        # Save to CSV (raw data only)
        with open(csv_filename, "w") as f:
            f.write("Time(s),Distance(m)\n")
            for t, d in zip(recorded_times, recorded_distances_m):
                f.write(f"{t:.2f},{d:.4f}\n")
        print(f"[INFO] CSV saved: {csv_filename}")

        # Save plot with raw data only
        plt.figure()
        x = np.array(recorded_times)
        y = np.array(recorded_distances_m)

        plt.plot(x, y, label='Raw Distance', color='tab:blue', linewidth=2.0)
        plt.xlabel("Time (s)")
        plt.ylabel("Distance (m)")
        plt.title("Recorded Distance")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(plot_filename)
        print(f"[INFO] Plot saved: {plot_filename}")
        plt.show()

    elif event.key == 'q':
        print("[INFO] Resetting live graph...")
        timestamps.clear()
        distances_m.clear()
        rolling_window.clear()
        line_dist.set_data([], [])
        ax.relim()
        ax.autoscale_view()
        plt.draw()


# === Connect Key Events ===
fig.canvas.mpl_connect('key_press_event', on_key)

# === Start Animation ===
ani = FuncAnimation(fig, update, interval=100, blit=True, cache_frame_data=False)

try:
    plt.show()
except KeyboardInterrupt:
    print("Stopped.")
finally:
    ser.close()