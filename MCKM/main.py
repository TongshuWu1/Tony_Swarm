import pygame
from environment import draw_border, WORLD_WIDTH_METERS, WORLD_HEIGHT_METERS
from robot import Robot
from visualize import *
from landmark import Landmark
from util import init_pygame, check_button_click
from behavior import Behavior, Exploration  # Import the new autonomous behavior

# Initialize pygame and screen
screen = init_pygame()
pygame.display.set_caption("2D Robot Simulation")

# Initialize Robots and Behaviors
robots = [
    Robot(WORLD_WIDTH_METERS / 3, WORLD_HEIGHT_METERS * 2 / 3, motor_distance=0.5),
    # Robot(WORLD_WIDTH_METERS * 2 / 3, WORLD_HEIGHT_METERS * 2 / 3, motor_distance=0.5)
]

behaviors = [Behavior(robot) for robot in robots]

time_scale = 1.0  # Default time speed (1x normal speed)
time_scale_step = 0.5  # How much to increase/decrease speed per click
min_time_scale = 1.0  # Minimum speed (20% of normal speed)
max_time_scale = 15.0  # Maximum speed (3x normal speed)
path_update_interval = 0.2  # Plot path every 0.2 seconds
path_time_accumulators = [0 for _ in robots]  # Separate accumulators for each robot

buttons = create_buttons(screen.get_width(), screen.get_height())

# Define Landmarks
landmarks = [
    Landmark(4, 10, shape="circle", color="yellow"),
    Landmark(7.5, 12, shape="square", color="yellow"),
    Landmark(11, 10, shape="triangle", color="yellow"),
    Landmark(4, 30, shape="circle", color="orange"),
    Landmark(7.5, 28, shape="square", color="orange"),
    Landmark(11, 30, shape="triangle", color="orange"),
]
visible_landmarks = []
FOV_ANGLE = 80
VIEW_DISTANCE = 12
robot_paths = [[] for _ in robots]

clock = pygame.time.Clock()
running = True
paused = True  # Track pause state
mode = 1
elapsed_time = 0

while running:
    screen.fill((255, 255, 255))
    dt = (clock.tick(60) / 1000) * time_scale

    # Handle Events
    for event in pygame.event.get():
        if event.type == pygame.QUIT or pygame.key.get_pressed()[pygame.K_ESCAPE]:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            mouse_pos = pygame.mouse.get_pos()
            if check_button_click(buttons["start"], mouse_pos):
                paused = False
            elif check_button_click(buttons["pause"], mouse_pos):
                paused = True
            elif check_button_click(buttons["restart"], mouse_pos):
                robots = [
                    Robot(WORLD_WIDTH_METERS / 3, WORLD_HEIGHT_METERS * 2 / 3, motor_distance=0.5),
                    Robot(WORLD_WIDTH_METERS * 2 / 3, WORLD_HEIGHT_METERS * 2 / 3, motor_distance=0.5)
                ]

                behaviors = [Behavior(robot) for robot in robots]
                robot_paths = [[] for _ in robots]
                path_time_accumulators = [0 for _ in robots]  # Reset accumulators
                paused = True
                elapsed_time = 0  # Reset elapsed time
            elif check_button_click(buttons["speed_up"], mouse_pos):
                time_scale = min(time_scale + time_scale_step, max_time_scale)
            elif check_button_click(buttons["slow_down"], mouse_pos):
                time_scale = max(time_scale - time_scale_step, min_time_scale)
            elif check_button_click(buttons["behavior"], mouse_pos):  # Toggle behavior on button click
                if mode == 1:
                    behaviors = [Exploration(robot) for robot in robots]
                    mode = 2
                else:
                    behaviors = [Behavior(robot) for robot in robots]
                    mode = 1

    if not paused:
        elapsed_time += dt
        for i, robot in enumerate(robots):
            # Update robot movement with delta time
            robot.update(dt)
            behaviors[i].update(dt)  # Pass dt to behavior
            path_time_accumulators[i] += dt  # Accumulate elapsed time for each robot
            if path_time_accumulators[i] >= path_update_interval:
                robot_paths[i].append((robot.x, robot.y))  # Store robot position
                path_time_accumulators[i] = 0  # Reset accumulator for this robot
            # Detect landmarks inside FOV
            visible_landmarks = robot.detect_landmarks(landmarks, FOV_ANGLE, VIEW_DISTANCE)

    # Draw elements
    draw_border(screen)
    draw_landmarks(screen, visible_landmarks)  # Draw only visible landmarks
    for path in robot_paths:
        draw_path(screen, path)
    for robot in robots:
        draw_robot(screen, robot)  # Ensure this uses the updated position
        draw_fov(screen, robot.x, robot.y, robot.angle, FOV_ANGLE, VIEW_DISTANCE)
    draw_buttons(screen, buttons)

    # Display motor speeds
    draw_motor_speeds(screen, robots[0].left_motor_speed, robots[0].right_motor_speed)
    draw_detected_landmarks(screen, visible_landmarks)
    draw_timer(screen, elapsed_time)
    draw_time_speed(screen, time_scale)
    pygame.display.flip()

pygame.quit()