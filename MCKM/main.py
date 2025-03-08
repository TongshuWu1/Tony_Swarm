import pygame
from environment import draw_border, WORLD_WIDTH_METERS, WORLD_HEIGHT_METERS
from robot import Robot
from visualize import *
from landmark import Landmark
from util import init_pygame

# Initialize pygame and screen
screen = init_pygame()
pygame.display.set_caption("2D Robot Simulation")

# Initialize Robot
robot = Robot(WORLD_WIDTH_METERS / 2, WORLD_HEIGHT_METERS / 2, motor_distance=0.4)

# Define Landmarks
landmarks = [
    Landmark(4, 10, shape="circle", color="yellow"),
    Landmark(7.5, 12, shape="square", color="yellow"),
    Landmark(11, 10, shape="triangle", color="yellow"),
    Landmark(4, 30, shape="circle", color="orange"),
    Landmark(7.5, 28, shape="square", color="orange"),
    Landmark(11, 30, shape="triangle", color="orange"),
]

FOV_ANGLE = 80
VIEW_DISTANCE = 12
robot_path = []

clock = pygame.time.Clock()
running = True

while running:
    screen.fill((255, 255, 255))
    dt = clock.tick(60) / 1000

    robot_path.append((robot.x, robot.y))

    # Handle key inputs for movement while allowing turning
    keys = pygame.key.get_pressed()

    left_speed = 0.0
    right_speed = 0.0
    full_speed = robot.max_speed
    half_speed = robot.max_speed / 2

    moving_forward = keys[pygame.K_UP]
    moving_backward = keys[pygame.K_DOWN]
    turning_left = keys[pygame.K_LEFT]
    turning_right = keys[pygame.K_RIGHT]

    if moving_forward:
        left_speed = full_speed
        right_speed = full_speed
    elif moving_backward:
        left_speed = -full_speed
        right_speed = -full_speed

    if turning_left:
        if moving_forward:
            left_speed = half_speed  # Reduce left motor speed
        elif moving_backward:
            right_speed = -half_speed  # Reduce right motor speed (for smooth backward turning)
        else:
            left_speed = -half_speed
            right_speed = half_speed

    elif turning_right:
        if moving_forward:
            right_speed = half_speed  # Reduce right motor speed
        elif moving_backward:
            left_speed = -half_speed  # Reduce left motor speed (for smooth backward turning)
        else:
            left_speed = half_speed
            right_speed = -half_speed

    # Apply motor speeds
    robot.set_motor_speeds(left_speed=left_speed, right_speed=right_speed)
    robot.update(dt)

    # Detect landmarks inside FOV
    visible_landmarks = robot.detect_landmarks(landmarks, FOV_ANGLE, VIEW_DISTANCE)

    # Draw elements
    draw_border(screen)
    draw_landmarks(screen, visible_landmarks)
    draw_path(screen, robot_path, step=7)
    draw_robot(screen, robot)
    draw_fov(screen, robot.x, robot.y, robot.angle, FOV_ANGLE, VIEW_DISTANCE)

    # Display motor speeds
    draw_motor_speeds(screen, robot.left_motor_speed, robot.right_motor_speed)

    # Display detected landmarks
    draw_detected_landmarks(screen, visible_landmarks)

    # Exit condition
    for event in pygame.event.get():
        if event.type == pygame.QUIT or keys[pygame.K_ESCAPE]:
            running = False

    pygame.display.flip()

pygame.quit()
