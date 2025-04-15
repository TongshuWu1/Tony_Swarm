# behavior.py
import numpy as np
from config import WALL_DETECTION_DISTANCE, WORLD_WIDTH_METERS, WORLD_HEIGHT_METERS

class BasicBehavior:
    def __init__(self, robot, estimator):
        self.robot = robot
        self.estimator = estimator
        self.turning = False

    def update_behavior(self):
        x, y, theta = self.estimator.mu
        angle_rad = np.radians(theta)
        lookahead_x = x + WALL_DETECTION_DISTANCE * np.cos(angle_rad)
        lookahead_y = y + WALL_DETECTION_DISTANCE * np.sin(angle_rad)

        wall_detected = (
            lookahead_x < 0 or lookahead_x > WORLD_WIDTH_METERS or
            lookahead_y < 0 or lookahead_y > WORLD_HEIGHT_METERS
        )

        if wall_detected:
            self.turning = True
            vl, vr = 1.5, 0.5
        else:
            self.turning = False
            vl = vr = 1.5

        return vl, vr
