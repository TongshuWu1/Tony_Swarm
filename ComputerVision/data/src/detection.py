import sensor, image, time, math

sampling_position = "off"  # "left", "center", "right", or "off"

R_GAIN = 84  # Red gain
G_GAIN = 64  # Green gain
B_GAIN = 88  # Blue gain
# Run get_gains.py with the camera and point the envronment to balance RGB gain of the camera so it keeps the gaussian info balanced through different lighting conditions.

SENSITIVITY =2.7
CONFIDENCE_THRESHOLD = 0.35 * (1.0 / SENSITIVITY)
# Higher = less strict color match, Lower = more strict. 1.5 to 3 is a good range

DECAY = 0.7
# Confidence map decay: lower = confident decrease faster from time

MAX_COV_TRACE = 400
# Tracker reset if uncertainty exceeds this

DT = 0.025  # Time between frames, change according to FPS, Im getting 8 fps so around 0.025 is a good dt
Q = 3.0     # Motion noise of balloon
R = 10.0    # Measurement noise
# Higher Q = trust in motion, Higher R = trust in detection. current value is good, if a higher noise is wanted, increase Q

VEL_ALPHA = 0.6
VEL_DECAY = 0.98
# VEL_ALPHA = smoothing for velocity

POS_SMOOTH_ALPHA = 0.8
# Smoothing for displayed position (cross marker)

def mat_add(A, B): return [[A[i][j] + B[i][j] for j in range(len(A[0]))] for i in range(len(A))]
def mat_sub(A, B): return [[A[i][j] - B[i][j] for j in range(len(A[0]))] for i in range(len(A))]
def mat_mul(A, B): return [[sum(A[i][k]*B[k][j] for k in range(len(B))) for j in range(len(B[0]))] for i in range(len(A))]
def mat_transpose(A): return [list(i) for i in zip(*A)]
def mat_identity(n): return [[1 if i == j else 0 for j in range(n)] for i in range(n)]
def mat_inv_2x2(m):
    det = m[0][0]*m[1][1] - m[0][1]*m[1][0]
    return [[ m[1][1]/det, -m[0][1]/det],
            [-m[1][0]/det,  m[0][0]/det]]

class KalmanTracker:
    def __init__(self):
        self.X = [[0], [0], [0], [0]]
        self.P = mat_identity(4)
        self.tracking = False
        self.A = [[1, 0, DT, 0],
                  [0, 1, 0, DT],
                  [0, 0, 1,  0],
                  [0, 0, 0,  1]]
        self.H = [[1, 0, 0, 0],
                  [0, 1, 0, 0]]
        self.Q = [[Q if i == j else 0 for j in range(4)] for i in range(4)]
        self.R = [[R, 0], [0, R]]
        self.I = mat_identity(4)
        self.sx = None
        self.sy = None

    def predict(self):
        self.X = mat_mul(self.A, self.X)
        At = mat_transpose(self.A)
        self.P = mat_add(mat_mul(mat_mul(self.A, self.P), At), self.Q)
        self.X[2][0] *= VEL_DECAY
        self.X[3][0] *= VEL_DECAY

        # Smooth displayed position
        x, y = self.X[0][0], self.X[1][0]
        if self.sx is None:
            self.sx, self.sy = x, y
        else:
            self.sx = POS_SMOOTH_ALPHA * x + (1 - POS_SMOOTH_ALPHA) * self.sx
            self.sy = POS_SMOOTH_ALPHA * y + (1 - POS_SMOOTH_ALPHA) * self.sy

        if self.P[0][0] + self.P[1][1] > MAX_COV_TRACE:
            self.tracking = False
            self.X = [[0], [0], [0], [0]]
            self.P = mat_identity(4)

    def correct(self, x, y):
        Z = [[x], [y]]
        Ht = mat_transpose(self.H)
        S = mat_add(mat_mul(mat_mul(self.H, self.P), Ht), self.R)
        K = mat_mul(mat_mul(self.P, Ht), mat_inv_2x2(S))
        Y = mat_sub(Z, mat_mul(self.H, self.X))
        prev_x, prev_y = self.X[0][0], self.X[1][0]
        self.X = mat_add(self.X, mat_mul(K, Y))
        self.P = mat_mul(mat_sub(self.I, mat_mul(K, self.H)), self.P)

        vx_obs = (self.X[0][0] - prev_x) / DT
        vy_obs = (self.X[1][0] - prev_y) / DT
        self.X[2][0] = VEL_ALPHA * vx_obs + (1 - VEL_ALPHA) * self.X[2][0]
        self.X[3][0] = VEL_ALPHA * vy_obs + (1 - VEL_ALPHA) * self.X[3][0]
        self.tracking = True

    def get_prediction(self): return int(self.sx), int(self.sy)
    def get_uncertainty_radius(self): return int(math.sqrt(self.P[0][0] + self.P[1][1]))
    def is_tracking(self): return self.tracking

def init_balloon_sensor(R_GAIN, G_GAIN, B_GAIN, framesize=sensor.HQVGA, board='NICLA'):
    sensor.reset()
    sensor.set_auto_whitebal(True)
    sensor.set_auto_exposure(True)
    sensor.set_pixformat(sensor.RGB565)
    if board == 'NICLA':
        sensor.ioctl(sensor.IOCTL_SET_FOV_WIDE, True)
    sensor.set_framesize(framesize)
    sensor.set_auto_whitebal(False)
    sensor.set_auto_exposure(False)
    if board == 'NICLA':
        sensor.__write_reg(0xfe, 0)
        sensor.__write_reg(0x80, 0b01111110)
        sensor.__write_reg(0x81, 0b01010100)
        sensor.__write_reg(0x82, 0b00000100)
        sensor.__write_reg(0x9a, 0b00001111)
        sensor.skip_frames(2)
        sensor.__write_reg(0xfe, 2)
        sensor.__write_reg(0x90, 0b11101101)
        sensor.__write_reg(0x91, 0b11000000)
        sensor.__write_reg(0x96, 0b00001100)
        sensor.__write_reg(0x97, 0x88)
        sensor.__write_reg(0x9b, 0b00100010)
        sensor.skip_frames(2)
        sensor.__write_reg(0xfe, 1)
        sensor.__write_reg(0x9a, 0b11110111)
        sensor.__write_reg(0x9d, 0xff)
        sensor.skip_frames(2)
        sensor.set_auto_exposure(False)
        sensor.set_auto_whitebal(False)
        sensor.__write_reg(0xb3, 64)
        sensor.__write_reg(0xb4, 64)
        sensor.__write_reg(0xb5, 64)
        sensor.__write_reg(0xfe, 0)
        sensor.__write_reg(0xad, int(R_GAIN))
        sensor.__write_reg(0xae, int(G_GAIN))
        sensor.__write_reg(0xaf, int(B_GAIN))
        sensor.set_auto_exposure(True)
        sensor.__write_reg(0xfe, 1)
        sensor.__write_reg(0x13, 120)
        sensor.skip_frames(2)
        sensor.__write_reg(0xfe, 2)
        sensor.__write_reg(0xd0, 72)
        sensor.__write_reg(0xd1, 56)
        sensor.__write_reg(0xd2, 56)
        sensor.__write_reg(0xd3, 40)
        sensor.__write_reg(0xd5, 0)

# LAB sampling
def run_ab_sampling():
    img = sensor.snapshot()
    IMG_WIDTH = img.width()
    IMG_HEIGHT = img.height()
    N_ROWS, N_COLS = 21, 28
    CELL_W, CELL_H = IMG_WIDTH // N_COLS, IMG_HEIGHT // N_ROWS
    CENTER_ROW = N_ROWS // 2
    CENTER_COL = {"left": N_COLS // 6, "center": N_COLS // 2, "right": (5 * N_COLS) // 6}[sampling_position]
    START_ROW, START_COL = CENTER_ROW - 1, CENTER_COL - 1

    def get_avg_ab(img, x, y, w, h):
        stats = img.get_statistics(roi=(x, y, w, h), thresholds=[(0, 255)])
        return round(stats.a_mean()), round(stats.b_mean())

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

def run_detection():
    GAUSSIAN_INFO = '[44.66631998613278, 25.87034148032588] , [[0.051555602012378035, -0.04802673024667995], [-0.04802673024667995, 0.06352078640446661]]'
    mean, cov = eval(GAUSSIAN_INFO.split(', [[')[0]), eval('[[' + GAUSSIAN_INFO.split(', [[')[1])

    def gaussian_confidence(a, b, mean, cov):
        da = a - mean[0]
        db = b - mean[1]
        m = da * (cov[0][0] * da + cov[0][1] * db) + db * (cov[1][0] * da + cov[1][1] * db)
        return 1 / (1 + 0.5 * m + 0.125 * m * m)

    def get_avg_ab(img, x, y, w, h):
        stats = img.get_statistics(roi=(x, y, w, h))
        return stats.a_mean(), stats.b_mean()

    tracker = KalmanTracker()
    IMG_WIDTH = sensor.width()
    IMG_HEIGHT = sensor.height()
    N_ROWS, N_COLS = 12, 18
    CELL_W = IMG_WIDTH // N_COLS
    CELL_H = IMG_HEIGHT // N_ROWS
    cell_confidence = [[0.0 for _ in range(N_COLS)] for _ in range(N_ROWS)]
    clock = time.clock()

    while True:
        clock.tick()
        img = sensor.snapshot()
        mask = image.Image(IMG_WIDTH, IMG_HEIGHT, sensor.GRAYSCALE)
        mask.clear()

        best_blob = None
        best_score = 0

        for row in range(N_ROWS):
            for col in range(N_COLS):
                x = col * CELL_W
                y = row * CELL_H
                a, b = get_avg_ab(img, x, y, CELL_W, CELL_H)
                conf = gaussian_confidence(a, b, mean, cov)
                if conf > CONFIDENCE_THRESHOLD:
                    cell_confidence[row][col] = min(cell_confidence[row][col] + conf, 1.0)
                    cell_confidence[row][col] = min(cell_confidence[row][col] + conf, 1.0)
                    mask.draw_rectangle(x, y, CELL_W, CELL_H, color=255, fill=True)
                    # # Draw border (no fill)
                    # img.draw_rectangle(x, y, CELL_W, CELL_H, color=(0, 255, 0), fill=False)
                else:
                    cell_confidence[row][col] *= DECAY

        mask.binary([(127, 255)])
        mask.close(2)
        mask.open(2)

        # Predict before evaluating blobs
        tracker.predict()
        tracker_cx, tracker_cy = tracker.get_prediction() if tracker.is_tracking() else (-1000, -1000)

        for blob in mask.find_blobs([(127, 255)], pixels_threshold=50, area_threshold=50, merge=True):
            if 0.5 <= blob.w() / blob.h() <= 2.0 and blob.solidity() >= 0.4 and blob.elongation() <= 2.0:
                x, y, w, h = blob.rect()
                avg_conf = 0.0
                count = 0
                for r in range(y // CELL_H, min(N_ROWS, (y + h) // CELL_H + 1)):
                    for c in range(x // CELL_W, min(N_COLS, (x + w) // CELL_W + 1)):
                        avg_conf += cell_confidence[r][c]
                        count += 1
                if count > 0:
                    avg_conf /= count

                score = blob.area() * avg_conf

                dist = math.sqrt((blob.cx() - tracker_cx)**2 + (blob.cy() - tracker_cy)**2)
                if dist < 20:  # pixels
                    score *= 1.8  # Apply a boost to nearby blob

                if score > best_score:
                    best_blob = blob
                    best_score = score

        if best_blob:
            tracker.correct(best_blob.cx(), best_blob.cy())
            img.draw_circle(best_blob.cx(), best_blob.cy(),
                            max(best_blob.w(), best_blob.h()) // 2,
                            color=(255, 0, 0))

        if tracker.is_tracking():
            cx, cy = tracker.get_prediction()
            r = tracker.get_uncertainty_radius()
            img.draw_cross(cx, cy, color=(255, 255, 255))
            img.draw_circle(cx, cy, r, color=(255, 255, 0))

            row = max(0, min(N_ROWS - 1, cy // CELL_H))
            col = max(0, min(N_COLS - 1, cx // CELL_W))
            for dr in range(-1, 2):
                for dc in range(-1, 2):
                    rr, cc = row + dr, col + dc
                    if 0 <= rr < N_ROWS and 0 <= cc < N_COLS:
                        cell_confidence[rr][cc] = min(cell_confidence[rr][cc] + 1.0, 1.0)

        img.draw_string(2, 2, "FPS: %.1f" % clock.fps(), color=(255, 255, 255))

        # Draw gridline
        # for row in range(1, N_ROWS):
        #     y = row * CELL_H
        #     img.draw_line(0, y, IMG_WIDTH, y, color=(64, 64, 64))
        # for col in range(1, N_COLS):
        #     x = col * CELL_W
        #     img.draw_line(x, 0, x, IMG_HEIGHT, color=(64, 64, 64))


if __name__ == "__main__":
    FRAME_SIZE = sensor.HQVGA
    board = "NICLA"
    clock = time.clock()
    sensor.reset()
    sensor.set_pixformat(sensor.RGB565)
    if board == "NICLA":
        sensor.ioctl(sensor.IOCTL_SET_FOV_WIDE, True)
    sensor.set_framesize(FRAME_SIZE)
    sensor.skip_frames(time=1000)
    sensor.snapshot()
    init_balloon_sensor(R_GAIN, G_GAIN, B_GAIN, framesize=FRAME_SIZE, board=board)
    sensor.skip_frames(time=1000)
    sensor.snapshot()
    if sampling_position in ("left", "center", "right"):
        run_ab_sampling()
    else:
        run_detection()
