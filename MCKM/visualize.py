import pygame
import math
from environment import SCALE, BORDER_LEFT, BORDER_TOP

def draw_robot(screen, robot):
    """ Draws a pointier triangle representing the robot at (x, y) with direction angle """
    x = BORDER_LEFT + robot.x * SCALE  # Convert meters to pixel position
    y = BORDER_TOP + robot.y * SCALE

    # Ensure robot size is **always the same real-world size** in pixels
    size = robot.size * SCALE  # Convert meters to pixels

    # Define triangle points relative to the robot's center
    p1 = (x + math.cos(math.radians(robot.angle)) * size, y - math.sin(math.radians(robot.angle)) * size)
    p2 = (x + math.cos(math.radians(robot.angle + 140)) * (size * 0.65), y - math.sin(math.radians(robot.angle + 140)) * (size * 0.65))
    p3 = (x + math.cos(math.radians(robot.angle - 140)) * (size * 0.65), y - math.sin(math.radians(robot.angle - 140)) * (size * 0.65))

    pygame.draw.polygon(screen, (200, 0, 0), [p1, p2, p3])

def draw_landmarks(screen, landmarks):
    """ Draws all landmarks on the screen """
    for landmark in landmarks:
        x = BORDER_LEFT + landmark.x * SCALE  # Convert meters to pixels
        y = BORDER_TOP + landmark.y * SCALE
        size = landmark.size * SCALE  # Convert meters to pixels

        if landmark.shape == "circle":
            pygame.draw.circle(screen, landmark.color, (int(x), int(y)), int(size / 2))

        elif landmark.shape == "square":
            pygame.draw.rect(screen, landmark.color, (int(x - size / 2), int(y - size / 2), int(size), int(size)))

        elif landmark.shape == "triangle":
            p1 = (x, y - size / 2)
            p2 = (x - size / 2, y + size / 2)
            p3 = (x + size / 2, y + size / 2)
            pygame.draw.polygon(screen, landmark.color, [p1, p2, p3])

def draw_fov(screen, robot_x, robot_y, robot_angle, fov_angle, view_distance):
    """ Draws the robot's field of view as a cone """
    x = BORDER_LEFT + robot_x * SCALE  # Convert meters to pixels
    y = BORDER_TOP + robot_y * SCALE

    fov_half_angle = fov_angle / 2
    angle_left = robot_angle - fov_half_angle
    angle_right = robot_angle + fov_half_angle

    # Calculate the points for the FOV cone
    p1 = (x + math.cos(math.radians(angle_left)) * view_distance * SCALE, y - math.sin(math.radians(angle_left)) * view_distance * SCALE)
    p2 = (x + math.cos(math.radians(angle_right)) * view_distance * SCALE, y - math.sin(math.radians(angle_right)) * view_distance * SCALE)

    # Draw FOV as a cone (triangle)
    pygame.draw.polygon(screen, (0, 255, 0), [ (x, y), p1, p2 ])  # Green FOV cone

def draw_path(screen, path, step=7):
    """ Draws a continuous path showing the full movement history of the robot """
    if len(path) < 2:
        return  # No path to draw yet

    # Convert path points to screen pixels
    pixel_path = [(BORDER_LEFT + x * SCALE, BORDER_TOP + y * SCALE) for x, y in path]

    # Draw dots instead of a continuous line
    for i in range(0, len(pixel_path), step):
        pygame.draw.circle(screen, (0, 0, 255), pixel_path[i], 1)  # Larger blue dot for better resolution


def draw_fov(screen, robot_x, robot_y, robot_angle, fov_angle, view_distance):
    """ Draws the robot's field of view as a cone with a circular base, pointing forward. """
    x = BORDER_LEFT + robot_x * SCALE  # Convert meters to pixels
    y = BORDER_TOP + robot_y * SCALE

    fov_half_angle = fov_angle / 2

    # Calculate the correct left and right boundary angles (relative to the robot's forward direction)
    angle_left = robot_angle + fov_half_angle
    angle_right = robot_angle - fov_half_angle

    # Calculate the points for the FOV cone outline
    p1 = (x + math.cos(math.radians(angle_left)) * view_distance * SCALE,
          y - math.sin(math.radians(angle_left)) * view_distance * SCALE)

    p2 = (x + math.cos(math.radians(angle_right)) * view_distance * SCALE,
          y - math.sin(math.radians(angle_right)) * view_distance * SCALE)

    # Draw the outline of the FOV as a cone
    pygame.draw.line(screen, (0, 255, 0), (x, y), p1, 2)  # Left side of the cone
    pygame.draw.line(screen, (0, 255, 0), (x, y), p2, 2)  # Right side of the cone

    # Draw the circular base of the cone
    pygame.draw.arc(screen, (0, 255, 0),
                    (x - view_distance * SCALE, y - view_distance * SCALE,
                     2 * view_distance * SCALE, 2 * view_distance * SCALE),
                    math.radians(angle_right), math.radians(angle_left), 1)

def draw_motor_speeds(screen, left_speed, right_speed):
    """ Display the current motor speeds on the screen. """
    font = pygame.font.Font(None, 28)  # Use default font, size 28
    text_color = (0, 0, 0)  # Black text

    # Create text surfaces
    left_text = font.render(f"Left Motor Speed: {left_speed:.2f}", True, text_color)
    right_text = font.render(f"Right Motor Speed: {right_speed:.2f}", True, text_color)

    # Display text on the top-left corner
    screen.blit(left_text, (10, 10))
    screen.blit(right_text, (10, 40))

def draw_detected_landmarks(screen, detected_landmarks):
    """ Display the currently detected landmarks in a box on the right side. """
    font = pygame.font.Font(None, 24)
    text_color = (0, 0, 0)  # Black text
    box_color = (200, 200, 200)  # Light gray box
    border_color = (0, 0, 0)  # Black border

    box_x = screen.get_width() - 220
    box_y = 10
    box_width = 200
    box_height = 150  # Adjust based on content

    # Draw the background box
    pygame.draw.rect(screen, box_color, (box_x, box_y, box_width, box_height))
    pygame.draw.rect(screen, border_color, (box_x, box_y, box_width, box_height), 2)

    # Title
    title_text = font.render("Landmarks in View", True, text_color)
    screen.blit(title_text, (box_x + 10, box_y + 10))

    # List detected landmarks
    y_offset = 40
    if detected_landmarks:
        for landmark in detected_landmarks:
            text = f"{landmark.color_name.capitalize()} {landmark.shape}"
            text_surface = font.render(text, True, text_color)
            screen.blit(text_surface, (box_x + 10, box_y + y_offset))
            y_offset += 25  # Spacing between entries
    else:
        no_text = font.render("None", True, text_color)
        screen.blit(no_text, (box_x + 10, box_y + y_offset))