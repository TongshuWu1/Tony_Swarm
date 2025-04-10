import serial
import struct
import time
import matplotlib.pyplot as plt
from collections import deque
import matplotlib.animation as animation

# Serial setup
ser = serial.Serial('/dev/cu.usbmodem11201', 115200, timeout=1)
print("Listening for pressure & distance...")

# Data storage
x_counter = 0
x_vals = deque(maxlen=100)           # Use numeric x-axis (frame count)
distances = deque(maxlen=100)
pressures = deque(maxlen=100)
roll_distances = deque(maxlen=100)
roll_pressures = deque(maxlen=100)

ROLLING_WINDOW = 1

def rolling_average(data, window):
    if len(data) < window:
        return sum(data) / len(data)
    return sum(list(data)[-window:]) / window

# Plot setup
fig, ax = plt.subplots()
line_dist, = ax.plot([], [], label='Distance (cm)', color='tab:blue')
line_press, = ax.plot([], [], label='Pressure (g)', color='tab:orange')
line_roll_dist, = ax.plot([], [], label='Rolling Avg Distance', linestyle='--', color='blue')
line_roll_press, = ax.plot([], [], label='Rolling Avg Pressure', linestyle='--', color='orange')

ax.set_xlabel("Sample Index")
ax.set_ylabel("Value")
ax.legend(loc='upper left')
plt.tight_layout()

distance = None
pressure = None

def update(frame):
    global distance, pressure, x_counter

    raw = ser.read(3)
    if len(raw) != 3:
        return line_dist, line_press, line_roll_dist, line_roll_press

    dtype = chr(raw[0])
    value = struct.unpack('<H', raw[1:])[0]
    timestamp = time.strftime('%H:%M:%S', time.localtime())

    if dtype == 'D':
        distance = value
    elif dtype == 'P':
        pressure = value

    if distance is not None and pressure is not None:
        x_vals.append(x_counter)
        x_counter += 1
        distances.append(distance)
        pressures.append(pressure)
        roll_distances.append(rolling_average(distances, ROLLING_WINDOW))
        roll_pressures.append(rolling_average(pressures, ROLLING_WINDOW))

        # Console output
        print(f"[{timestamp}] Distance: {distance} cm | Pressure: {pressure} g")

        # Update plots
        line_dist.set_data(x_vals, distances)
        line_press.set_data(x_vals, pressures)
        line_roll_dist.set_data(x_vals, roll_distances)
        line_roll_press.set_data(x_vals, roll_pressures)

        ax.relim()
        ax.autoscale_view()

    return line_dist, line_press, line_roll_dist, line_roll_press

ani = animation.FuncAnimation(fig, update, interval=200, blit=True, cache_frame_data=False)

try:
    plt.show()
except KeyboardInterrupt:
    print("Stopped.")
finally:
    ser.close()
