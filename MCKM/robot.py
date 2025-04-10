import math
import numpy as np
from environment import WORLD_WIDTH_METERS, WORLD_HEIGHT_METERS

class Robot:
    def __init__(self, x, y, angle=90, size=0.5, motor_distance=0.5, max_speed=1.0, noise_std=(0.1, 0.1, 0.05)):
        self.x = x
        self.y = y
        self.angle = angle  # Degrees, 0 points to the right, 90 up

        self.size = size
        self.motor_distance = motor_distance
        self.max_speed = max_speed

        self.left_motor_speed = 0.0
        self.right_motor_speed = 0.0

        self.velocity = 0.0
        self.angular_velocity = 0.0

        self.noise_std = np.array(noise_std)
        self.last_visible_landmarks = set()

    def set_motor_speeds(self, left_speed=0.0, right_speed=0.0):
        self.left_motor_speed = max(-self.max_speed, min(self.max_speed, left_speed))
        self.right_motor_speed = max(-self.max_speed, min(self.max_speed, right_speed))

    def set_velocity(self, linear_velocity, angular_velocity):
        L = self.motor_distance
        vl = linear_velocity - (angular_velocity * L / 2)
        vr = linear_velocity + (angular_velocity * L / 2)
        self.set_motor_speeds(vl, vr)

    def update(self, dt):
        vl = self.left_motor_speed
        vr = self.right_motor_speed
        L = self.motor_distance

        self.velocity = (vl + vr) / 2
        self.angular_velocity = (vr - vl) / L

        # Add noise to the robot's motion
        v_noisy = self.velocity + np.random.normal(0, self.noise_std[0])
        omega_noisy = self.angular_velocity + np.random.normal(0, self.noise_std[2])

        theta = math.radians(self.angle)
        delta_x = v_noisy * math.cos(theta) * dt
        delta_y = v_noisy * math.sin(theta) * dt

        new_x = self.x + delta_x
        new_y = self.y + delta_y

        # Boundary enforcement
        half_size = self.size / 2
        self.x = max(half_size, min(WORLD_WIDTH_METERS - half_size, new_x))
        self.y = max(half_size, min(WORLD_HEIGHT_METERS - half_size, new_y))

        # Orientation update
        self.angle = (self.angle + math.degrees(omega_noisy * dt)) % 360

    def detect_landmarks(self, landmarks, fov_angle, view_distance):
        detected_landmarks = set()

        for landmark in landmarks:
            if self.is_landmark_within_fov(landmark, fov_angle, view_distance):
                detected_landmarks.add((landmark.shape, landmark.color_name, landmark.x, landmark.y))

        if detected_landmarks != self.last_visible_landmarks:
            self.last_visible_landmarks = detected_landmarks

        return [l for l in landmarks if (l.shape, l.color_name, l.x, l.y) in detected_landmarks]

    def is_landmark_within_fov(self, landmark, fov_angle, view_distance):
        dx = landmark.x - self.x
        dy = landmark.y - self.y
        distance = math.sqrt(dx ** 2 + dy ** 2)

        if distance > view_distance:
            return False

        landmark_angle = math.degrees(math.atan2(dy, dx)) % 360
        robot_angle = self.angle % 360

        fov_half = fov_angle / 2
        left_bound = (robot_angle - fov_half) % 360
        right_bound = (robot_angle + fov_half) % 360

        if left_bound < right_bound:
            return left_bound <= landmark_angle <= right_bound
        else:
            return landmark_angle >= left_bound or landmark_angle <= right_bound