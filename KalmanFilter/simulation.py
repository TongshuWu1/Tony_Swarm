import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
import numpy as np
import math

from config import *
from landmark import Landmark
from robot import Robot
from localization import OdometryEstimator
from behavior import BasicBehavior
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.patches import Wedge
from matplotlib.widgets import Button

class Simulator:
    def __init__(self):
        self.robot = Robot(x=7.5, y=20)
        self.fig, self.ax = plt.subplots(figsize=(7, 12))
        plt.subplots_adjust(right=0.85)
        self.ax.set_xlim(-1, WORLD_WIDTH_METERS + 1)
        self.ax.set_ylim(-1, WORLD_HEIGHT_METERS + 1)
        self.ax.set_aspect('equal')
        self.ax.set_title("Robot Simulator")
        self.ax.set_xlabel("X (meters)")
        self.ax.set_ylabel("Y (meters)")

        self.fov_angle = FOV_ANGLE
        self.view_distance = VIEW_DISTANCE

        self.inset_ax = inset_axes(self.ax, width="25%", height="25%",
                                   loc='center left',
                                   bbox_to_anchor=(1.02, 0.5, 1, 1),
                                   bbox_transform=self.ax.transAxes,
                                   borderpad=2)

        self.ax.add_patch(patches.Rectangle((0, 0), WORLD_WIDTH_METERS, WORLD_HEIGHT_METERS,
                                            linewidth=2, edgecolor='red', facecolor='none'))

        self.landmarks = [Landmark(**lm) for lm in LANDMARKS]
        for lm in self.landmarks:
            lm.draw(self.ax)

        self.path = PATH_WAYPOINTS
        self.current_target_index = 0
        self.auto_mode = True

        for (x, y) in self.path:
            self.ax.plot(x, y, marker='x', color='red', markersize=10, linewidth=2)

        self.robot_patch = patches.Polygon(self.robot_shape(), fc='blue', ec='black', zorder=10)
        self.ax.add_patch(self.robot_patch)

        init_cov = np.diag([0.1, 0.1, 1])
        self.odometry = OdometryEstimator(
            init_pos=[self.robot.x, self.robot.y, self.robot.angle],
            init_cov=init_cov
        )

        self.behavior = BasicBehavior(self.robot, self.odometry)
        self.uncertainty_ellipse = None
        self.fov_wedge = None

        self.manual_forward = 0.0
        self.manual_turn = 0.0

        # Trace and lookahead
        self.trace_x = [self.robot.x]
        self.trace_y = [self.robot.y]
        self.trace_line, = self.ax.plot(self.trace_x, self.trace_y, color='black', linewidth=1, zorder=0)
        self.lookahead_line = None
        self.trace_counter = 0
        self.trace_interval = 5

        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        self.fig.canvas.mpl_connect('key_release_event', self.on_key_release)

        reset_ax = self.fig.add_axes([0.82, 0.05, 0.12, 0.05])
        self.reset_button = Button(reset_ax, 'Reset')
        self.reset_button.on_clicked(self.reset_simulation)

        auto_ax = self.fig.add_axes([0.67, 0.05, 0.12, 0.05])
        self.auto_button = Button(auto_ax, 'Auto: ON')
        self.auto_button.on_clicked(self.toggle_auto_mode)

        self.state_text = self.ax.text(1.02, 0.3, "", transform=self.ax.transAxes,
                                       fontsize=10, verticalalignment='top', family='monospace')

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
        if not self.auto_mode:
            if event.key == 'up':
                self.manual_forward = 1.0
            elif event.key == 'down':
                self.manual_forward = -1.0
            elif event.key == 'left':
                self.manual_turn = 1.5
            elif event.key == 'right':
                self.manual_turn = -1.5

    def on_key_release(self, event):
        if not self.auto_mode:
            if event.key in ['up', 'down']:
                self.manual_forward = 0.0
            elif event.key in ['left', 'right']:
                self.manual_turn = 0.0

    def toggle_auto_mode(self, event=None):
        self.auto_mode = not self.auto_mode
        self.auto_button.label.set_text('Auto: ON' if self.auto_mode else 'Auto: OFF')
        self.current_target_index = 0
        print(f"[Toggle] Auto Mode is now {'ON' if self.auto_mode else 'OFF'}")

    def reset_simulation(self, event=None):
        self.robot.x = 7.5
        self.robot.y = 20
        self.robot.angle = 90
        self.robot.left_motor_speed = 0.0
        self.robot.right_motor_speed = 0.0

        init_cov = np.diag([0.1, 0.1, 1])
        self.odometry.mu = np.array([self.robot.x, self.robot.y, self.robot.angle], dtype=float)
        self.odometry.cov = init_cov
        self.current_target_index = 0
        self.trace_x = [self.robot.x]
        self.trace_y = [self.robot.y]
        self.trace_line.set_data(self.trace_x, self.trace_y)
        self.trace_counter = 0
        print("Simulation reset.")

    def update(self, frame):
        dt = TIME_STEP
        L = self.robot.motor_distance

        if self.auto_mode:
            x_real, y_real, theta_real = self.robot.x, self.robot.y, self.robot.angle
            angle_rad = math.radians(theta_real)
            fx = x_real + WALL_DETECTION_DISTANCE * math.cos(angle_rad)
            fy = y_real + WALL_DETECTION_DISTANCE * math.sin(angle_rad)

            wall = fx < 0 or fx > WORLD_WIDTH_METERS or fy < 0 or fy > WORLD_HEIGHT_METERS

            if wall:
                vl, vr = 0.5, 1.5
            else:
                x_est, y_est, theta_est = self.odometry.mu
                if self.current_target_index < len(self.path):
                    tx, ty = self.path[self.current_target_index]
                    dx = tx - x_est
                    dy = ty - y_est
                    dist = math.hypot(dx, dy)
                    target_angle = math.degrees(math.atan2(dy, dx)) % 360
                    angle_diff = (target_angle - theta_est + 540) % 360 - 180

                    v = 1.0 if dist > 0.3 else 0.0
                    omega = np.clip(angle_diff / 5.0, -3.0, 3.0)
                    vl = v - omega * L / 2
                    vr = v + omega * L / 2

                    if dist < 0.3:
                        self.current_target_index += 1
                        if LOOP_WAYPOINTS:
                            self.current_target_index %= len(self.path)
                else:
                    vl = vr = 0.0

            self.robot.set_motor_speeds(vl, vr)
        else:
            v = self.manual_forward
            omega = self.manual_turn
            vl = v - omega * L / 2
            vr = v + omega * L / 2
            self.robot.set_motor_speeds(vl, vr)

        self.robot.update(dt)

        vl = self.robot.left_motor_speed
        vr = self.robot.right_motor_speed
        linear_velocity = (vl + vr) / 2.0
        angular_velocity = (vr - vl) / L

        # Add noise to the velocities
        self.odometry.predict(
            [linear_velocity, angular_velocity],
            dt,
            np.diag([
                PREDICTION_NOISE[0],
                PREDICTION_NOISE[1],
                np.radians(PREDICTION_NOISE[2])**2
            ])
        )

        detected = self.robot.detect_landmarks(self.landmarks, self.fov_angle, self.view_distance)
        measurements = []
        for lm in detected:
            dx, dy = lm.x - self.robot.x, lm.y - self.robot.y
            r = np.hypot(dx, dy) + np.random.normal(0, 0.1)
            true_bearing = (np.degrees(np.arctan2(dy, dx)) - self.robot.angle) % 360
            if true_bearing > 180: true_bearing -= 360
            b = true_bearing + np.random.normal(0, 3)
            measurements.append((r, b, lm.x, lm.y))

        if measurements:
            self.odometry.correct(
                measurements,
                np.diag([MEASUREMENT_NOISE[0]**2, MEASUREMENT_NOISE[1]**2]),
                alpha=MEASUREMENT_ALPHA
            )

        self.robot_patch.set_xy(self.robot_shape())

        if self.uncertainty_ellipse:
            self.uncertainty_ellipse.remove()
        self.odometry.draw_uncertainty_ellipse(self.ax, n_std=3.0, edgecolor='cyan', facecolor='none', linewidth=2)
        self.uncertainty_ellipse = self.ax.patches[-1]

        if self.fov_wedge:
            self.fov_wedge.remove()
        theta1 = self.robot.angle - self.fov_angle / 2
        theta2 = self.robot.angle + self.fov_angle / 2
        self.fov_wedge = patches.Wedge((self.robot.x, self.robot.y), r=self.view_distance,
                                       theta1=theta1, theta2=theta2,
                                       facecolor='blue', alpha=0.2, edgecolor='blue', lw=1, zorder=3)
        self.ax.add_patch(self.fov_wedge)

        self.trace_counter += 1
        if self.trace_counter % self.trace_interval == 0:
            self.trace_x.append(self.robot.x)
            self.trace_y.append(self.robot.y)
            self.trace_line.set_data(self.trace_x, self.trace_y)

        lookahead_x = self.robot.x + WALL_DETECTION_DISTANCE * math.cos(math.radians(self.robot.angle))
        lookahead_y = self.robot.y + WALL_DETECTION_DISTANCE * math.sin(math.radians(self.robot.angle))

        if self.lookahead_line is None:
            self.lookahead_line, = self.ax.plot(
                [self.robot.x, lookahead_x], [self.robot.y, lookahead_y],
                color='magenta', linestyle='--', linewidth=1, zorder=1
            )
        else:
            self.lookahead_line.set_data([self.robot.x, lookahead_x], [self.robot.y, lookahead_y])

        self.inset_ax.clear()
        self.inset_ax.set_title("Detected Landmarks", fontsize=8)
        self.inset_ax.set_xticks([])
        self.inset_ax.set_yticks([])
        self.inset_ax.set_xlim(self.robot.x - self.view_distance, self.robot.x + self.view_distance)
        self.inset_ax.set_ylim(self.robot.y - self.view_distance, self.robot.y + self.view_distance)

        for lm in detected:
            color = 'yellow' if lm.color_name == 'yellow' else 'orange'
            shape = lm.shape
            if shape == 'circle':
                self.inset_ax.add_patch(patches.Circle((lm.x, lm.y), lm.size / 2, facecolor=color, edgecolor='black'))
            elif shape == 'square':
                self.inset_ax.add_patch(patches.Rectangle((lm.x - lm.size / 2, lm.y - lm.size / 2),
                                                          lm.size, lm.size, facecolor=color, edgecolor='black'))
            elif shape == 'triangle':
                half = lm.size / 2
                pts = [(lm.x, lm.y + half), (lm.x - half, lm.y - half), (lm.x + half, lm.y - half)]
                self.inset_ax.add_patch(patches.Polygon(pts, facecolor=color, edgecolor='black'))

        self.inset_ax.plot(self.robot.x, self.robot.y, 'bo', markersize=4)

        x, y, angle = self.odometry.mu
        self.state_text.set_text(f"Estimated State:\nX: {x:.2f}\nY: {y:.2f}\nθ: {angle:.1f}°")
        if self.current_target_index < len(self.path):
            tx, ty = self.path[self.current_target_index]
            self.state_text.set_text(self.state_text.get_text() + f"\nTarget: ({tx:.1f}, {ty:.1f})")
        else:
            self.state_text.set_text(self.state_text.get_text() + "\nTarget: None")

        return [self.robot_patch, self.uncertainty_ellipse, self.fov_wedge, self.lookahead_line, self.trace_line]

    def run(self):
        ani = FuncAnimation(self.fig, self.update, interval=50, blit=False, frames=500)
        plt.show()

if __name__ == "__main__":
    sim = Simulator()
    sim.run()
