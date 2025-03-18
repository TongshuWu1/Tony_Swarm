import math
import numpy as np
from environment import WORLD_WIDTH_METERS, WORLD_HEIGHT_METERS

class Robot:
    def __init__(self, x, y, angle=90, size=0.5, motor_distance=0.5, max_speed=1.0):
        self.x = x
        self.y = y
        self.angle = angle  # Degrees

        self.size = size
        self.motor_distance = motor_distance  # Distance between motors (meters)
        self.max_speed = max_speed  # Maximum speed per motor

        # Differential drive motor speeds
        self.left_motor_speed = 0.0
        self.right_motor_speed = 0.0

        # **Fix: Track detected landmarks to avoid AttributeError**
        self.last_visible_landmarks = set()

        # **Kalman Filter Initialization**
        self.state = np.array([x, y, 0, 0])  # [x, y, vx, vy]
        self.P = np.eye(4) * 0.05  # Initial uncertainty

        # **Process noise covariance (adjusted for stability)**
        self.Q = np.array([
            [0.0005, 0, 0, 0],
            [0, 0.0005, 0, 0],
            [0, 0, 0.005, 0],
            [0, 0, 0, 0.005]
        ])

        # **Uncertainty Decay Factor (Balances Accuracy & Growth)**
        self.P_decay = 0.6

    def set_motor_speeds(self, left_speed=0.0, right_speed=0.0):
        """ Set individual speeds for each motor. Ensures speed stays within allowed limits. """
        self.left_motor_speed = max(-self.max_speed, min(self.max_speed, left_speed))
        self.right_motor_speed = max(-self.max_speed, min(self.max_speed, right_speed))

    def update(self, dt):
        """ Updates Kalman Filter position and real-world movement based on motor speeds. """

        # **Compute actual movement from motor speeds**
        vl = self.left_motor_speed
        vr = self.right_motor_speed
        v = (vl + vr) / 2  # Forward velocity
        omega = (vr - vl) / self.motor_distance  # Angular velocity
        theta = math.radians(self.angle)

        # **Update Kalman Filter Prediction (State Estimation)**
        self.state[0] += v * math.cos(theta) * dt  # Predict x position
        self.state[1] -= v * math.sin(theta) * dt  # Predict y position (inverted in pygame)
        self.state[2] += omega * dt  # Predict x velocity


        # **Kalman Prediction Step**
        F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        self.P = F @ self.P @ F.T + self.Q  # Update covariance

        # # **Apply Decay** (Prevents infinite uncertainty growth)
        # self.P *= self.P_decay

        # **Update the real robot position based on Kalman Filter state**
        self.x = self.state[0]
        self.y = self.state[1]
        self.angle = (self.angle + math.degrees(omega * dt)) % 360  # Keep within 0-360

        # **Ensure robot stays inside world boundaries**
        self.x = np.clip(self.x, self.size / 2, WORLD_WIDTH_METERS - self.size / 2)
        self.y = np.clip(self.y, self.size / 2, WORLD_HEIGHT_METERS - self.size / 2)

    def detect_landmarks(self, landmarks, fov_angle, view_distance):
        """ Detects landmarks within the robot's FOV. Returns a list of detected landmarks. """
        detected_landmarks = set()
        for landmark in landmarks:
            if self.is_landmark_within_fov(landmark, fov_angle, view_distance):
                detected_landmarks.add((landmark.shape, landmark.color_name, landmark.x, landmark.y))

        # **Update landmark tracking**
        self.last_visible_landmarks = detected_landmarks

        return [landmark for landmark in landmarks if (landmark.shape, landmark.color_name, landmark.x, landmark.y) in detected_landmarks]

    def is_landmark_within_fov(self, landmark, fov_angle, view_distance):
        """ Returns True if the landmark is within the robot's field of view. """
        dx = landmark.x - self.x
        dy = landmark.y - self.y
        distance = math.sqrt(dx**2 + dy**2)

        if distance > view_distance:
            return False  # Too far away

        # **Compute angle from robot to landmark**
        landmark_angle = math.degrees(math.atan2(-dy, dx))  # -dy because of Pygame's inverted Y-axis
        landmark_angle = (landmark_angle + 360) % 360
        robot_angle = (self.angle + 360) % 360

        fov_half = fov_angle / 2
        left_boundary = (robot_angle - fov_half) % 360
        right_boundary = (robot_angle + fov_half) % 360

        if left_boundary < right_boundary:
            return left_boundary <= landmark_angle <= right_boundary
        else:
            return landmark_angle >= left_boundary or landmark_angle <= right_boundary
