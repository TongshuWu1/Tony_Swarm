import sensor, image, time

# === Choose which region to sample ===
sampling_position = "right"  # Change to "left", "center", or "right" manually

# === Sensor setup ===
sensor.reset()
sensor.set_pixformat(sensor.RGB565)
sensor.set_framesize(sensor.QVGA)  # 320x240
sensor.skip_frames(time=2000)

sensor.set_auto_whitebal(False)
sensor.set_auto_gain(True)
sensor.set_auto_exposure(True)

# === Grid setup ===
N_ROWS = 21
N_COLS = 28
IMG_WIDTH = 320
IMG_HEIGHT = 240

CELL_W = IMG_WIDTH // N_COLS
CELL_H = IMG_HEIGHT // N_ROWS


if sampling_position == "left":
    CENTER_COL = N_COLS // 6
    CENTER_ROW = N_ROWS // 2
elif sampling_position == "center":
    CENTER_COL = N_COLS // 2
    CENTER_ROW = N_ROWS // 2
elif sampling_position == "right":
    CENTER_COL = (5 * N_COLS) // 6  # farther right
    CENTER_ROW = (5 * N_ROWS) // 6  # slightly lower
else:
    raise ValueError("Invalid sampling position!")

START_ROW = CENTER_ROW - 1
START_COL = CENTER_COL - 1

# === Get LAB A/B ===
def get_avg_ab(img, x, y, w, h):
    stats = img.get_statistics(roi=(x, y, w, h), thresholds=[(0, 255)])
    a = stats.a_mean()
    b = stats.b_mean()
    return round(a), round(b)

# === Main loop ===
while True:
    img = sensor.snapshot()

    for row_offset in range(3):
        row = START_ROW + row_offset
        line = []
        for col_offset in range(3):
            col = START_COL + col_offset
            x = col * CELL_W
            y = row * CELL_H
            a, b = get_avg_ab(img, x, y, CELL_W, CELL_H)
            line.append(f"({a}, {b})")
            img.draw_rectangle(x, y, CELL_W, CELL_H, color=(255, 0, 0))
        print(", ".join(line) + ",")
        time.sleep_ms(20)
