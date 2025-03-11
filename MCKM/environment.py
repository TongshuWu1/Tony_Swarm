import pygame

# Initialize Pygame
pygame.init()

# Fixed screen size (canvas resolution)
SCREEN_WIDTH = 1300  # pixels
SCREEN_HEIGHT = 900  # pixels

# Define real-world workspace size (Configurable in meters)
WORLD_WIDTH_METERS = 15  # Change this to resize the border
WORLD_HEIGHT_METERS = 40  # Change this to resize the border

# Scaling factor: 1 meter = constant pixel ratio
SCALE = 22

# Compute the red border rectangle (scaled workspace inside the fixed canvas)
BORDER_LEFT = (SCREEN_WIDTH - WORLD_WIDTH_METERS * SCALE) // 2
BORDER_TOP = (SCREEN_HEIGHT - WORLD_HEIGHT_METERS * SCALE) // 2
BORDER_RIGHT = BORDER_LEFT + WORLD_WIDTH_METERS * SCALE
BORDER_BOTTOM = BORDER_TOP + WORLD_HEIGHT_METERS * SCALE

def draw_border(screen):
    """ Draws a red border representing the real-world workspace """
    pygame.draw.rect(screen, (200, 0, 0),
                     (BORDER_LEFT, BORDER_TOP, WORLD_WIDTH_METERS * SCALE, WORLD_HEIGHT_METERS * SCALE), 5)
