import pygame
from environment import SCREEN_WIDTH, SCREEN_HEIGHT

def init_pygame():
    """ Initializes pygame and creates a fixed-size screen. """
    pygame.init()
    return pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))

def normalize_angle(angle):
    """ Ensures the angle remains within [0, 360] degrees. """
    return angle % 360

def angle_difference(target, current):
    """ Returns the shortest signed angle difference. """
    diff = (target - current) % 360  # Normalize
    if diff > 180:
        diff -= 360  # Convert to the shortest path (-180 to 180)
    return diff


def check_button_click(button, mouse_pos):
    """ Checks if the mouse click is inside a button. """
    return button.collidepoint(mouse_pos)