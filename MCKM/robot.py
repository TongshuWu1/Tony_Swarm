import math
import numpy as np
from environment import SCALE, BORDER_LEFT, BORDER_TOP, WORLD_WIDTH_METERS, WORLD_HEIGHT_METERS

class Robot:
    def __init__(self, x, y, angle=0, size=0.5, motor_distance=0.4, max_speed=1.0):
        """
        - x, y: Initial position in meters
        - angle: Orientation in degrees
        - size: Robot size in meters
        - motor_distance: Distance between left and right motors
        - max_speed: Maximum motor speed (m/s)
        """
        self.x = x
        self.y = y
        self.angle = angle  # Degrees

        self.size = size
        self.motor_distance = motor_distance  # Distance between motors (meters)
        self.max_speed = max_speed  # Maximum speed per motor

        # Differential drive motor speeds
        self.left_motor_speed = 0.0
        self.right_motor_speed = 0.0

        # Landmark detection tracking
        self.last_visible_landmarks = set()

    def set_motor_speeds(self, left_speed=0.0, right_speed=0.0):
        """ Set individual speeds for each motor (defaults to 0 if not pressed). """
        self.left_motor_speed = max(-self.max_speed, min(self.max_speed, left_speed))
        self.right_motor_speed = max(-self.max_speed, min(self.max_speed, right_speed))

    def update(self, dt):
        """ Updates the robot's position based on differential drive kinematics and prevents boundary exit """
        vl = self.left_motor_speed  # Left motor velocity
        vr = self.right_motor_speed  # Right motor velocity
        L = self.motor_distance  # Distance between motors

        # Compute linear and angular velocity
        v = (vl + vr) / 2  # Forward velocity (m/s)
        omega = (vr - vl) / L  # Angular velocity (rad/s)

        # Predict new position
        theta = math.radians(self.angle)
        new_x = self.x + v * math.cos(theta) * dt
        new_y = self.y - v * math.sin(theta) * dt  # Negative y due to Pygame coordinates

        # Enforce boundary conditions
        min_x = 0 + self.size / 2
        max_x = WORLD_WIDTH_METERS - self.size / 2
        min_y = 0 + self.size / 2
        max_y = WORLD_HEIGHT_METERS - self.size / 2

        if min_x <= new_x <= max_x:
            self.x = new_x
        if min_y <= new_y <= max_y:
            self.y = new_y

        self.angle += math.degrees(omega * dt)
        self.angle %= 360  # Keep angle within [0, 360]

    def detect_landmarks(self, landmarks, fov_angle, view_distance):
        """ Detects landmarks within the robot's FOV and prints if the list changes """
        detected_landmarks = set()

        for landmark in landmarks:
            if self.is_landmark_within_fov(landmark, fov_angle, view_distance):
                detected_landmarks.add((landmark.shape, landmark.color_name, landmark.x, landmark.y))

        # If the visible landmarks have changed, print the new set
        if detected_landmarks != self.last_visible_landmarks:
            self.last_visible_landmarks = detected_landmarks
            if detected_landmarks:
                print("\**Current Landmarks in View:**")
                for shape, color, x, y in detected_landmarks:
                    print(f" - {color.capitalize()} {shape} at ({x:.2f}, {y:.2f})")
            else:
                print("\nNo landmarks in view")

        return [landmark for landmark in landmarks if (landmark.shape, landmark.color_name, landmark.x, landmark.y) in detected_landmarks]

    def is_landmark_within_fov(self, landmark, fov_angle, view_distance):
        """ Returns True if the landmark is fully within the FOV """
        dx = landmark.x - self.x
        dy = landmark.y - self.y
        distance = math.sqrt(dx**2 + dy**2)

        if distance > view_distance:
            return False  # Too far away

        # Compute angle from robot to landmark
        landmark_angle = math.degrees(math.atan2(-dy, dx))  # -dy because Pygame y-axis is inverted
        landmark_angle = (landmark_angle + 360) % 360
        robot_angle = (self.angle + 360) % 360

        fov_half = fov_angle / 2
        left_boundary = (robot_angle - fov_half) % 360
        right_boundary = (robot_angle + fov_half) % 360

        if left_boundary < right_boundary:
            return left_boundary <= landmark_angle <= right_boundary
        else:
            return landmark_angle >= left_boundary or landmark_angle <= right_boundary
