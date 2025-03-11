import random
import math
from util import normalize_angle, angle_difference
from robot import Robot
from environment import WORLD_WIDTH_METERS, WORLD_HEIGHT_METERS

# Configurable Parameters
TURN_ANGLE_RANGE = (90, 180)  # Right-hand turn between 90 and 180 degrees
SPIRAL_TIME_RANGE = (10, 20)  # Spiral runs between 15s and 25s
FORWARD_TIME_RANGE = (4,15)  # Forward motion lasts between 8s and 14s
ULTRASONIC_DISTANCE = 0.4
STRAIGHT_WALK_TIME = 3  # After turning, move straight for 3 seconds

class Behavior:
    def __init__(self, robot: Robot):
        self.robot = robot
        self.turning = False
        self.target_angle = None
        self.forward_timer = random.uniform(*FORWARD_TIME_RANGE)
        self.straight_timer = 0

    def update(self, dt):
        if self.turning:
            self._handle_turn(dt)
        elif self.straight_timer > 0:
            self._move_straight_after_turn(dt)
        else:
            self._move_forward(dt)

    def _move_forward(self, dt):
        if self.forward_timer > 0:
            self.robot.set_motor_speeds(self.robot.max_speed, self.robot.max_speed)
            self.forward_timer -= dt
            if self._near_wall():
                self._handle_wall_avoidance()
        else:
            self._choose_random_angle()

    def _near_wall(self):
        buffer = self.robot.size
        scan_x = self.robot.x + math.cos(math.radians(self.robot.angle)) * ULTRASONIC_DISTANCE
        scan_y = self.robot.y - math.sin(math.radians(self.robot.angle)) * ULTRASONIC_DISTANCE
        return (
            scan_x - buffer <= 0 or scan_x + buffer >= WORLD_WIDTH_METERS or
            scan_y - buffer <= 0 or scan_y + buffer >= WORLD_HEIGHT_METERS
        )

    def _handle_wall_avoidance(self):
        if not self.turning:
            self.turning = True
            self.target_angle = normalize_angle(self.robot.angle + random.uniform(*TURN_ANGLE_RANGE))
            # print(f"[WALL] Detected! Turning right to {self.target_angle:.2f}°")

    def _handle_turn(self, dt):
        angle_diff = angle_difference(self.target_angle, self.robot.angle)
        turn_speed = self.robot.max_speed / 2
        angular_velocity = turn_speed / self.robot.motor_distance
        delta_angle = math.degrees(angular_velocity * dt)
        # print(f"[TURN] Current: {self.robot.angle:.2f}, Target: {self.target_angle:.2f}, Diff: {angle_diff:.2f}")

        if abs(angle_diff) <= delta_angle:
            self.robot.angle = self.target_angle
            self.turning = False
            self.straight_timer = STRAIGHT_WALK_TIME
            # print("[TURN] Completed. Moving straight for 3 seconds.")
            return

        self.robot.set_motor_speeds(turn_speed, -turn_speed)

    def _move_straight_after_turn(self, dt):
        if self.straight_timer > 0:
            self.robot.set_motor_speeds(self.robot.max_speed, self.robot.max_speed)
            self.straight_timer -= dt
        else:
            # print("[MOVE] Finished straight movement. Entering spiral mode.")
            self._start_spiral_movement()

    def _choose_random_angle(self):
        self.target_angle = normalize_angle(self.robot.angle + random.uniform(*TURN_ANGLE_RANGE))
        self.turning = True
        # print(f"[EXPLORE] Choosing random turn angle: {self.target_angle:.2f}°")

class Exploration(Behavior):
    def __init__(self, robot: Robot):
        super().__init__(robot)
        self.spiral_timer = random.uniform(*SPIRAL_TIME_RANGE)
        self.initial_angular_velocity = 1.2
        self.mode = "spiral"
        self.choosing_angle = False

    def update(self, dt):
        if self.turning:
            self._handle_turn(dt)
        elif self.choosing_angle:
            self._choose_random_angle()
        elif self.straight_timer > 0:
            self._move_straight_after_turn(dt)
        elif self.mode == "spiral":
            self._spiral_movement(dt)
        else:
            self._move_forward(dt)

    def _spiral_movement(self, dt):
        if self.spiral_timer <= 0:
            self._start_forward_movement()
            return

        self.initial_angular_velocity -= 0.02 * dt
        linear_velocity = self.robot.max_speed
        angular_velocity = max(0.1, self.initial_angular_velocity) / self.robot.motor_distance
        self.robot.set_velocity(linear_velocity, angular_velocity)
        self.spiral_timer -= dt

        if self._near_wall():
            self._handle_wall_avoidance()

    def _handle_turn(self, dt):
        super()._handle_turn(dt)
        if not self.turning:
            self._start_spiral_movement()

    def _start_spiral_movement(self):
        self.spiral_timer = random.uniform(*SPIRAL_TIME_RANGE)
        self.initial_angular_velocity = 1.0
        self.mode = "spiral"

    def _start_forward_movement(self):
        self.forward_timer = random.uniform(*FORWARD_TIME_RANGE)
        self.mode = "forward"
