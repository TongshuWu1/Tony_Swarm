# config.py

WORLD_WIDTH_METERS = 15
WORLD_HEIGHT_METERS = 40

LANDMARKS = [
    {'x': 4, 'y': 10, 'shape': 'circle', 'color': 'yellow', 'size': 1.1},
    {'x': 7.5, 'y': 12, 'shape': 'square', 'color': 'yellow', 'size': 1.1},
    {'x': 11, 'y': 10, 'shape': 'triangle', 'color': 'yellow', 'size': 1.1},
    {'x': 4, 'y': 30, 'shape': 'circle', 'color': 'orange', 'size': 1.1},
    {'x': 7.5, 'y': 28, 'shape': 'square', 'color': 'orange', 'size': 1.1},
    {'x': 11, 'y': 30, 'shape': 'triangle', 'color': 'orange', 'size': 1.1},
]

TIME_STEP = 0.1
TOTAL_TIME = 100  # Increased for interactive control

# Field of View Settings
FOV_ANGLE = 80        # degrees
VIEW_DISTANCE = 8    # meters
