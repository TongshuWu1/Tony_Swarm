import pygame
import math
from environment import SCALE, BORDER_LEFT, BORDER_TOP

class Landmark:
    """ Represents a static landmark in the environment. """
    def __init__(self, x, y, shape="circle", color="yellow", size=1.1):
        """
        - x, y: Position in meters (real-world coordinates)
        - shape: "circle", "triangle", or "square"
        - color: "yellow" or "orange"
        - size: Landmark size in meters
        """
        self.x = x
        self.y = y
        self.shape = shape
        self.color_name = color  # Store the color name as a string
        self.color = (255, 204, 0) if color == "yellow" else (255, 140, 0)  # Store the RGB value
        self.size = size  # Size in meters

    def draw(self, screen):
        """ Draw the landmark on the screen at the correct position. """
        x = BORDER_LEFT + self.x * SCALE  # Convert meters to pixels
        y = BORDER_TOP + self.y * SCALE
        size = self.size * SCALE  # Convert meters to pixels

        if self.shape == "circle":
            pygame.draw.circle(screen, self.color, (int(x), int(y)), int(size / 2))

        elif self.shape == "square":
            pygame.draw.rect(screen, self.color, (int(x - size / 2), int(y - size / 2), int(size), int(size)))

        elif self.shape == "triangle":
            p1 = (x, y - size / 2)
            p2 = (x - size / 2, y + size / 2)
            p3 = (x + size / 2, y + size / 2)
            pygame.draw.polygon(screen, self.color, [p1, p2, p3])
