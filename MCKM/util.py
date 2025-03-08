import pygame
from environment import SCREEN_WIDTH, SCREEN_HEIGHT

def init_pygame():
    """ Initializes pygame and creates a fixed-size screen. """
    pygame.init()
    return pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
