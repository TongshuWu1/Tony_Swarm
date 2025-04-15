import os
import serial
import struct
import time
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.animation import FuncAnimation
from datetime import datetime, timedelta
from collections import deque
import numpy as np
from scipy.signal import savgol_filter

# === Config ===
PORT = '/dev/cu.usbmodem11101'  # Update your serial port
BAUD = 115200
READ_INTERVAL = 0.1
SPIKE_THRESHOLD = 1
ROLLING_WINDOW_SIZE = 10

# === Serial Setup ===
ser = serial.Serial(PORT, BAUD, timeout=1)
print("Listening for distance data...")

# === Data Storage ===
timestamps = []
distances_m = []
recording = False
recorded_timestamps = []
recorded_distances_m = []
last_read_time = 0
rolling_window = deque(maxlen=ROLLING_WINDOW_SIZE)
last_saved_dt = None
last_saved_value = None

# === Plot Setup ===
fig, ax = plt.subplots()
line_dist, = ax.plot([], [], label='Distance (m)', color='tab:blue', linewidth=1.5, alpha=0.9)

ax.set_xlabel("Time")
ax.set_ylabel("Distance (m)")
ax.set_title("Live Distance from TFmini")
ax.legend(loc='upper left')
ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S.%f'))
ax.xaxis.set_major_locator(mdates.AutoDateLocator())
plt.xticks(rotation=45)
plt.tight_layout()

# === Live Update Function ===
def update(frame):
    global recording, last_read_time, last_saved_dt, last_saved_value

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

        if len(rolling_window) >= 3:
            median = np.median(rolling_window)
            if abs(distance_m - median) > SPIKE_THRESHOLD:
                print(f"[WARN] Spike filtered: {distance_m:.2f} m (median: {median:.2f})")
                continue
        rolling_window.append(distance_m)

        now_dt = datetime.fromtimestamp(time.time()).replace(microsecond=(datetime.now().microsecond // 1000) * 1000)

        if last_saved_dt is not None:
            gap = int((now_dt - last_saved_dt).total_seconds() * 1000)
            if gap == 0:
                continue
            elif gap > 1:
                for i in range(1, gap):
                    interp_time = last_saved_dt + timedelta(milliseconds=i)
                    interp_value = last_saved_value + (distance_m - last_saved_value) * (i / gap)
                    timestamps.append(interp_time)
                    distances_m.append(interp_value)
                    if recording:
                        ts_str = interp_time.strftime("%H:%M:%S.%f")[:-3]
                        recorded_timestamps.append(ts_str)
                        recorded_distances_m.append(interp_value)

        timestamps.append(now_dt)
        distances_m.append(distance_m)
        last_saved_dt = now_dt
        last_saved_value = distance_m

        ts_str = now_dt.strftime("%H:%M:%S.%f")[:-3]
        print(f"[{ts_str}] Distance: {distance_m:.2f} m")
        if recording:
            recorded_timestamps.append(ts_str)
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
    global recording, timestamps, distances_m, rolling_window, recorded_timestamps, recorded_distances_m, last_saved_dt, last_saved_value

    if event.key == 'r' and not recording:
        print("[INFO] Started recording...")
        recording = True
        recorded_timestamps.clear()
        recorded_distances_m.clear()

    elif event.key == 't' and recording:
        print("[INFO] Stopping and saving recording...")
        recording = False

        folder_name = "weight853_orange"
        os.makedirs(folder_name, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_filename = os.path.join(folder_name, f"recorded_distance_{timestamp}.csv")
        plot_filename = os.path.join(folder_name, f"recorded_distance_plot_{timestamp}.png")

        # Save to CSV
        with open(csv_filename, "w") as f:
            f.write("Timestamp,Distance(m)\n")
            for t, d in zip(recorded_timestamps, recorded_distances_m):
                f.write(f"{t},{d:.4f}\n")
        print(f"[INFO] CSV saved: {csv_filename}")

        # === Sudden Increase Detection ===
        print("[INFO] Detecting sudden increase in distance...")
        time_array = np.array([
            (datetime.strptime(t, "%H:%M:%S.%f") - datetime.strptime(recorded_timestamps[0], "%H:%M:%S.%f")).total_seconds()
            for t in recorded_timestamps
        ])
        distance_array = np.array(recorded_distances_m)

        window_size = min(51, len(distance_array) // 2 * 2 + 1)
        filtered_distance = savgol_filter(distance_array, window_length=window_size, polyorder=2)

        delta_distance = np.diff(filtered_distance)
        delta_time = np.diff(time_array)
        rate_of_change = np.insert(delta_distance / delta_time, 0, 0)

        window = 300
        max_diff = 0
        max_change_idx = -1

        for i in range(window, len(rate_of_change) - window):
            prev_slope = np.mean(rate_of_change[i - window:i])
            next_slope = np.mean(rate_of_change[i:i + window])
            slope_diff = next_slope - prev_slope  # Only upward jumps

            if slope_diff > max_diff and slope_diff > 0:
                max_diff = slope_diff
                max_change_idx = i

        change_time = time_array[max_change_idx]
        change_distance = distance_array[max_change_idx]
        print(f"[INFO] Max sudden increase: {change_distance:.3f} m at t = {change_time:.3f} s")

        # === Plot with annotation ===
        plt.figure()
        plt.plot(time_array, distance_array, label="Raw", alpha=0.5)
        plt.plot(time_array, filtered_distance, label="Filtered", linewidth=2.0)
        plt.axvline(change_time, color='red', linestyle='--', label=f'Sudden increase: {change_distance:.2f} m')
        plt.xlabel("Time (s)")
        plt.ylabel("Distance (m)")
        plt.title("Recorded Distance with Sudden Increase")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(plot_filename)
        print(f"[INFO] Plot saved: {plot_filename}")
        plt.show()

    elif event.key == 'y':
        print("[INFO] Resetting all data and refreshing graph...")
        timestamps.clear()
        distances_m.clear()
        rolling_window.clear()
        recorded_timestamps.clear()
        recorded_distances_m.clear()
        last_saved_dt = None
        last_saved_value = None
        line_dist.set_data([], [])
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
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
