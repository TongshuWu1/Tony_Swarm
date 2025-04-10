# robot.py

import math
import numpy as np
from config import WORLD_WIDTH_METERS, WORLD_HEIGHT_METERS


class Robot:
    def __init__(self, x, y, angle=90, size=0.5, motor_distance=0.5, max_speed=2.0, noise_std=(0.02, 0.02, 0.2)):
        """
        Differential-drive robot simulation with motion noise and landmark perception.

        Parameters:
          x, y: Initial position (meters)
          angle: Initial orientation in degrees (0 points right, 90 up)
          size: Robot size (meters)
          motor_distance: Distance between motors (meters)
          max_speed: Maximum speed (m/s)
          noise_std: Standard deviations for [x, y, angle] noise.
        """
        self.x = x
        self.y = y
        self.angle = angle  # in degrees
        self.size = size
        self.motor_distance = motor_distance
        self.max_speed = max_speed

        self.left_motor_speed = 0.0
        self.right_motor_speed = 0.0

        # Motion noise: [σ_x (m), σ_y (m), σ_angle (degrees)]
        self.noise_std = np.array(noise_std)

        # For landmark detection: remember last visible set (can be used for filtering)
        self.last_visible_landmarks = set()

    def set_motor_speeds(self, left_speed=0.0, right_speed=0.0):
        """Set motor speeds, enforcing limits."""
        self.left_motor_speed = np.clip(left_speed, -self.max_speed, self.max_speed)
        self.right_motor_speed = np.clip(right_speed, -self.max_speed, self.max_speed)

    def set_velocity(self, linear_velocity, angular_velocity):
        """
        Set linear and angular velocity by converting to left/right motor speeds.

        linear_velocity: m/s
        angular_velocity: rad/s
        """
        L = self.motor_distance
        vl = linear_velocity - (angular_velocity * L / 2)
        vr = linear_velocity + (angular_velocity * L / 2)
        self.set_motor_speeds(vl, vr)

    def update(self, dt):
        """
        Update the robot's state (position and orientation) with noise.
        """
        vl = self.left_motor_speed
        vr = self.right_motor_speed
        L = self.motor_distance

        # Compute the ideal linear and angular velocities.
        velocity = (vl + vr) / 2.0
        angular_velocity = (vr - vl) / L

        # Add noise to the velocities.
        v_noisy = velocity + np.random.normal(0, self.noise_std[0])
        omega_noisy = angular_velocity + np.random.normal(0, self.noise_std[2])

        theta_rad = math.radians(self.angle)
        delta_x = v_noisy * math.cos(theta_rad) * dt
        delta_y = v_noisy * math.sin(theta_rad) * dt

        new_x = self.x + delta_x
        new_y = self.y + delta_y

        # Enforce boundaries (assuming robot has a square footprint)
        half_size = self.size / 2
        self.x = max(half_size, min(WORLD_WIDTH_METERS - half_size, new_x))
        self.y = max(half_size, min(WORLD_HEIGHT_METERS - half_size, new_y))

        # Update orientation (convert omega from rad/s to degrees/s)
        self.angle = (self.angle + math.degrees(omega_noisy * dt)) % 360

    def detect_landmarks(self, landmarks, fov_angle, view_distance):
        """
        Detect landmarks within the robot's field-of-view (FOV) and view distance.

        Parameters:
          landmarks: list of landmark objects.
          fov_angle: Field of view in degrees.
          view_distance: Maximum distance (meters) at which landmarks are visible.

        Returns:
          A list of landmark objects detected within the FOV.
        """
        detected_landmarks = set()

        for landmark in landmarks:
            if self.is_landmark_within_fov(landmark, fov_angle, view_distance):
                detected_landmarks.add((landmark.shape, landmark.color_name, landmark.x, landmark.y))

        self.last_visible_landmarks = detected_landmarks

        # Return the actual landmark objects that match the detected ones.
        return [l for l in landmarks if (l.shape, l.color_name, l.x, l.y) in detected_landmarks]

    def is_landmark_within_fov(self, landmark, fov_angle, view_distance):
        """
        Check if a given landmark is within the robot's field-of-view and within view distance.

        Returns:
          True if the landmark is within the FOV and view distance; False otherwise.
        """
        dx = landmark.x - self.x
        dy = landmark.y - self.y
        distance = math.sqrt(dx ** 2 + dy ** 2)

        if distance > view_distance:
            return False

        # Compute the angle from the robot to the landmark.
        landmark_angle = math.degrees(math.atan2(dy, dx)) % 360
        robot_angle = self.angle % 360

        # Determine the left and right bounds of the FOV.
        fov_half = fov_angle / 2.0
        left_bound = (robot_angle - fov_half) % 360
        right_bound = (robot_angle + fov_half) % 360

        # Check if the landmark's angle lies within the FOV.
        if left_bound < right_bound:
            return left_bound <= landmark_angle <= right_bound
        else:
            # FOV spans the 0-degree boundary.
            return landmark_angle >= left_bound or landmark_angle <= right_bound
