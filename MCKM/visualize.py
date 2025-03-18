import pygame
import math
from environment import SCALE, BORDER_LEFT, BORDER_TOP
import numpy as np

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

def draw_path(screen, path):
    """ Draws a continuous path showing the full movement history of the robot. """
    if len(path) < 2:
        return  # No path to draw yet

    # Convert path points to screen pixels
    pixel_path = [(BORDER_LEFT + x * SCALE, BORDER_TOP + y * SCALE) for x, y in path]

    # Draw small circles at each saved path position
    for point in pixel_path:
        pygame.draw.lines(screen, (0, 0, 255), False, pixel_path, 1)



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


def create_buttons(screen_width, screen_height):
    """ Creates button positions aligned in two columns at the bottom left. """
    button_width = 120
    button_height = 40
    button_spacing = 15
    button_x_left = 30  # Left column position
    button_x_right = 180  # Right column position (aligned next to left column)
    button_y_start = screen_height - 3 * (button_height + button_spacing)  # Position near bottom

    buttons = {
        # Left Column (Simulation Controls)
        "start": pygame.Rect(button_x_left, button_y_start, button_width, button_height),
        "pause": pygame.Rect(button_x_left, button_y_start + (button_height + button_spacing), button_width, button_height),
        "restart": pygame.Rect(button_x_left, button_y_start + 2 * (button_height + button_spacing), button_width, button_height),

        # Right Column (Time Speed Controls)
        "speed_up": pygame.Rect(button_x_right, button_y_start, button_width, button_height),
        "slow_down": pygame.Rect(button_x_right, button_y_start + (button_height + button_spacing), button_width, button_height),

        # New Button for Changing Behavior
        "behavior": pygame.Rect(button_x_right, button_y_start + 2 * (button_height + button_spacing), button_width, button_height)
    }

    return buttons

def draw_buttons(screen, buttons):
    """ Draws buttons with a structured left-aligned two-column layout. """
    font = pygame.font.Font(None, 30)

    button_colors = {
        "start": (100, 200, 100),      # Green for Start
        "pause": (200, 100, 100),      # Red for Pause
        "restart": (100, 100, 200),    # Blue for Restart
        "speed_up": (0, 150, 255),     # Light Blue for Speed Up
        "slow_down": (255, 150, 0),    # Orange for Slow Down
        "behavior": (150, 0, 150)  # Purple for Change Behavior
    }

    for name, button in buttons.items():
        pygame.draw.rect(screen, button_colors[name], button)
        text = font.render(name.replace("_", " ").title(), True, (255, 255, 255))
        text_rect = text.get_rect(center=(button.x + button.width // 2, button.y + button.height // 2))
        screen.blit(text, text_rect)


def draw_timer(screen, elapsed_time):
    """ Display the simulation timer in the top-left corner. """
    font = pygame.font.Font(None, 28)
    text_color = (0, 0, 0)  # Black text
    timer_text = font.render(f"Time: {elapsed_time:.2f} s", True, text_color)
    screen.blit(timer_text, (10, 70))  # Position below motor speeds

def draw_time_speed(screen, time_scale):
    """ Display the current time speed on the screen. """
    font = pygame.font.Font(None, 28)
    text_color = (0, 0, 0)  # Black text
    speed_text = font.render(f"Speed: {time_scale:.1f}x", True, text_color)
    screen.blit(speed_text, (10, 100))  # Position below the timer

    def draw_timer_input_box(screen, input_box, timer_input, active):
        font = pygame.font.Font(None, 32)
        box_color = (0, 128, 255) if input_box.collidepoint(pygame.mouse.get_pos()) or active else (100, 100, 100)
        pygame.draw.rect(screen, box_color, input_box, 2)
        txt_surface = font.render(timer_input, True, (0, 0, 0))
        screen.blit(txt_surface, (input_box.x + 5, input_box.y + 5))

    def draw_simulation_timer(screen, elapsed_time, time_limit):
        font = pygame.font.Font(None, 28)
        timer_text = font.render(f"Time: {elapsed_time:.2f}/{time_limit:.2f} s", True, (0, 0, 0))
        screen.blit(timer_text, (10, 130))


def draw_kalman_estimate(screen, robot):
    """ Draws the Kalman Filter estimated position and uncertainty ellipse """
    estimated_x = BORDER_LEFT + robot.state[0] * SCALE
    estimated_y = BORDER_TOP + robot.state[1] * SCALE

    # Draw estimated position (Red Dot)
    pygame.draw.circle(screen, (255, 0, 0), (int(estimated_x), int(estimated_y)), 3)

    # Draw uncertainty ellipse
    draw_uncertainty_ellipse(screen, robot)

def draw_uncertainty_ellipse(screen, robot):
    """ Draws an uncertainty ellipse representing Gaussian distribution around estimated position """
    estimated_x = BORDER_LEFT + robot.state[0] * SCALE
    estimated_y = BORDER_TOP + robot.state[1] * SCALE

    # Extract covariance matrix for position
    P_pos = robot.P[:2, :2]  # Extract only position covariance

    # Compute eigenvalues and eigenvectors for scaling the ellipse
    eigenvalues, eigenvectors = np.linalg.eigh(P_pos)

    eigenvalues = np.clip(eigenvalues, 0.0001, 0.05)

    # Convert eigenvalues to standard deviation for 95% confidence interval
    scale_x = 1.5 * math.sqrt(eigenvalues[0]) * SCALE
    scale_y = 1.5 * math.sqrt(eigenvalues[1]) * SCALE

    # Compute rotation angle
    angle = math.degrees(math.atan2(eigenvectors[1, 0], eigenvectors[0, 0]))

    # Create an ellipse surface
    ellipse_surface = pygame.Surface((scale_x * 2, scale_y * 2), pygame.SRCALPHA)
    pygame.draw.ellipse(ellipse_surface, (255, 0, 0, 100), (0, 0, scale_x * 2, scale_y * 2), 2)

    # Rotate and blit the ellipse at the estimated position
    rotated_ellipse = pygame.transform.rotate(ellipse_surface, -angle)
    screen.blit(rotated_ellipse, (estimated_x - scale_x, estimated_y - scale_y))


def draw_actual_vs_predicted(screen, robot):
    """ Draws a line connecting the actual position to the predicted position """
    true_x = BORDER_LEFT + robot.x * SCALE
    true_y = BORDER_TOP + robot.y * SCALE
    estimated_x = BORDER_LEFT + robot.state[0] * SCALE
    estimated_y = BORDER_TOP + robot.state[1] * SCALE

    pygame.draw.line(screen, (0, 0, 255), (true_x, true_y), (estimated_x, estimated_y), 2)