import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
import numpy as np
import math

from config import WORLD_WIDTH_METERS, WORLD_HEIGHT_METERS, TIME_STEP, LANDMARKS, FOV_ANGLE, VIEW_DISTANCE
from landmark import Landmark
from robot import Robot
from localization import OdometryEstimator
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.patches import Wedge

class Simulator:
    def __init__(self):
        self.robot = Robot(x=7.5, y=20)

        # Setup figure
        self.fig, self.ax = plt.subplots(figsize=(7, 12))
        self.ax.set_xlim(-1, WORLD_WIDTH_METERS + 1)
        self.ax.set_ylim(-1, WORLD_HEIGHT_METERS + 1)
        self.ax.set_aspect('equal')
        self.ax.set_title("Robot Simulator with Odometry & FOV")
        self.ax.set_xlabel("X (meters)")
        self.ax.set_ylabel("Y (meters)")

        # Add inset for landmark mini-map
        self.inset_ax = inset_axes(self.ax, width="30%", height="30%", loc='upper right')
        self.inset_ax.set_title("Detected Landmarks", fontsize=8)
        self.inset_ax.set_xticks([])
        self.inset_ax.set_yticks([])

        # Draw workspace boundary
        workspace_rect = patches.Rectangle((0, 0), WORLD_WIDTH_METERS, WORLD_HEIGHT_METERS,
                                           linewidth=2, edgecolor='red', facecolor='none')
        self.ax.add_patch(workspace_rect)

        # Landmarks
        self.landmarks = [Landmark(**lm) for lm in LANDMARKS]
        for lm in self.landmarks:
            lm.draw(self.ax)

        # Robot patch
        self.robot_patch = patches.Polygon(self.robot_shape(), fc='blue', ec='black', zorder=10)
        self.ax.add_patch(self.robot_patch)

        # Odometry setup
        init_cov = np.diag([0.1, 0.1, 1])
        self.odometry = OdometryEstimator(
            init_pos=[self.robot.x, self.robot.y, self.robot.angle],
            init_cov=init_cov
        )
        self.uncertainty_ellipse = None

        # FOV
        self.fov_angle = FOV_ANGLE
        self.view_distance = VIEW_DISTANCE
        self.fov_wedge = None
        self.fov_lines = []

        # Control
        self.speed = 0
        self.rotation = 0
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        self.fig.canvas.mpl_connect('key_release_event', self.on_key_release)

    def robot_shape(self):
        angle_rad = math.radians(self.robot.angle)
        front = (self.robot.x + self.robot.size * math.cos(angle_rad),
                 self.robot.y + self.robot.size * math.sin(angle_rad))
        left = (self.robot.x + self.robot.size * 0.6 * math.cos(angle_rad + 2.5),
                self.robot.y + self.robot.size * 0.6 * math.sin(angle_rad + 2.5))
        right = (self.robot.x + self.robot.size * 0.6 * math.cos(angle_rad - 2.5),
                 self.robot.y + self.robot.size * 0.6 * math.sin(angle_rad - 2.5))
        return [front, left, right]

    def on_key_press(self, event):
        if event.key == 'up':
            self.speed = 1.5
        elif event.key == 'down':
            self.speed = -1.5
        elif event.key == 'left':
            self.rotation = 2
        elif event.key == 'right':
            self.rotation = -2

    def on_key_release(self, event):
        if event.key in ['up', 'down']:
            self.speed = 0
        elif event.key in ['left', 'right']:
            self.rotation = 0

    def update(self, frame):
        dt = TIME_STEP
        L = self.robot.motor_distance

        # Compute motion inputs
        v = self.speed
        omega = self.rotation
        vl = v - (omega * L / 2)
        vr = v + (omega * L / 2)
        self.robot.set_motor_speeds(vl, vr)
        self.robot.update(dt)

        # Actual robot motion
        linear_velocity = (vl + vr) / 2.0
        angular_velocity = (vr - vl) / L

        # Drift-aware noise — include rotation-driven position noise
        min_drift = np.array([0.005, 0.005, np.radians(0.2) ** 2])
        v_effect = max(abs(linear_velocity), 0.01)
        omega_effect = max(abs(angular_velocity), 0.01)

        motion_noise_cov = np.diag([
            max((0.01 * v_effect) ** 2 + (0.01 * omega_effect) ** 2, min_drift[0]),
            max((0.02 * v_effect) ** 2 + (0.01 * omega_effect) ** 2, min_drift[1]),
            max((np.radians(1.5) * omega_effect) ** 2, min_drift[2])
        ])

        # === DEBUG: PRINT MOTION NOISE INFO ===
        print(f"[Frame {frame}]")
        print("Linear Velocity (v):", linear_velocity)
        print("Angular Velocity (w):", angular_velocity)
        print("Motion Noise Covariance:\n", motion_noise_cov)

        self.odometry.predict([linear_velocity, angular_velocity], dt, motion_noise_cov)
        # --- Correction Step from Detected Landmarks ---
        detected = self.robot.detect_landmarks(self.landmarks, self.fov_angle, self.view_distance)
        measurements = []
        for lm in detected:
            dx = lm.x - self.robot.x
            dy = lm.y - self.robot.y
            range_meas = np.hypot(dx, dy) + np.random.normal(0, 0.1)
            bearing_true = (np.degrees(np.arctan2(dy, dx)) - self.robot.angle) % 360
            if bearing_true > 180:
                bearing_true -= 360
            bearing_meas = bearing_true + np.random.normal(0, 3)  # 3 degree noise
            measurements.append((range_meas, bearing_meas, lm.x, lm.y))

        if measurements:
            R = np.diag([0.1 ** 2, 3.0 ** 2])  # range, bearing noise
            self.odometry.correct(measurements, R, alpha=0.5)
        # DEBUG: Covariance growth check
        print("Covariance Matrix (x/y):\n", self.odometry.cov[:2, :2])
        std_xy = np.sqrt(np.diag(self.odometry.cov[:2, :2]))
        print("Std Dev X/Y:", std_xy, "\n")

        # Update robot visual
        self.robot_patch.set_xy(self.robot_shape())

        # Update uncertainty ellipse
        if self.uncertainty_ellipse:
            self.uncertainty_ellipse.remove()
        self.odometry.draw_uncertainty_ellipse(
            self.ax, n_std=3.0,
            edgecolor='cyan', facecolor='none', linewidth=2
        )
        self.uncertainty_ellipse = self.ax.patches[-1]

        # FOV wedge
        if self.fov_wedge:
            self.fov_wedge.remove()
        theta1 = self.robot.angle - self.fov_angle / 2
        theta2 = self.robot.angle + self.fov_angle / 2
        self.fov_wedge = patches.Wedge(
            (self.robot.x, self.robot.y), r=self.view_distance,
            theta1=theta1, theta2=theta2,
            facecolor='blue', alpha=0.2, edgecolor='blue', lw=1, zorder=3
        )
        self.ax.add_patch(self.fov_wedge)

        # FOV lines
        for line in self.fov_lines:
            line.remove()
        self.fov_lines.clear()
        for angle_deg in [theta1, theta2]:
            angle_rad = math.radians(angle_deg)
            x_end = self.robot.x + self.view_distance * math.cos(angle_rad)
            y_end = self.robot.y + self.view_distance * math.sin(angle_rad)
            line, = self.ax.plot([self.robot.x, x_end], [self.robot.y, y_end],
                                 linestyle='--', color='blue', linewidth=1, zorder=4)
            self.fov_lines.append(line)

        # Inset mini-map
        self.inset_ax.clear()
        self.inset_ax.set_title("Detected Landmarks", fontsize=8)
        self.inset_ax.set_xticks([])
        self.inset_ax.set_yticks([])
        self.inset_ax.set_xlim(self.robot.x - self.view_distance, self.robot.x + self.view_distance)
        self.inset_ax.set_ylim(self.robot.y - self.view_distance, self.robot.y + self.view_distance)

        detected = self.robot.detect_landmarks(self.landmarks, self.fov_angle, self.view_distance)
        for lm in detected:
            color = 'yellow' if lm.color_name == 'yellow' else 'orange'
            if lm.shape == 'circle':
                self.inset_ax.add_patch(patches.Circle((lm.x, lm.y), lm.size / 2,
                                                       facecolor=color, edgecolor='black'))
            elif lm.shape == 'square':
                self.inset_ax.add_patch(patches.Rectangle(
                    (lm.x - lm.size / 2, lm.y - lm.size / 2),
                    lm.size, lm.size, facecolor=color, edgecolor='black'))
            elif lm.shape == 'triangle':
                half = lm.size / 2
                pts = [(lm.x, lm.y + half), (lm.x - half, lm.y - half), (lm.x + half, lm.y - half)]
                self.inset_ax.add_patch(patches.Polygon(pts, facecolor=color, edgecolor='black'))

        self.inset_ax.plot(self.robot.x, self.robot.y, 'bo', markersize=4)

        return [self.robot_patch, self.uncertainty_ellipse, self.fov_wedge] + self.fov_lines

    def run(self):
        ani = FuncAnimation(self.fig, self.update, interval=50, blit=False, frames=500)
        plt.show()


if __name__ == "__main__":
    sim = Simulator()
    sim.run()
