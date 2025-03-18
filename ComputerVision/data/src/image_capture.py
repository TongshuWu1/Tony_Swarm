import sensor, image, time, os

# Initialize the camera
sensor.reset()
sensor.set_pixformat(sensor.RGB565)  # Color images
sensor.set_framesize(sensor.QVGA)    # 320x240 resolution
sensor.skip_frames(time=2000)        # Allow the camera to adjust

# Set save directory inside Nicla Vision's internal storage
SAVE_DIR = "/flash/data"

# Ensure the directory exists
try:
    os.mkdir(SAVE_DIR)
except OSError:
    pass  # Directory already exists

# Image capture parameters
NUM_IMAGES = 230 # Capture 5 images
INTERVAL_MS = 2000  # Capture every 3 seconds
counterNumber = 201  # Start file numbering from 1

clock = time.clock()  # Initialize FPS tracking

while counterNumber <= NUM_IMAGES:
    clock.tick()  # Keep FPS tracking active
    img = sensor.snapshot()  # Capture a frame and keep buffer running

    # Save the image while keeping the live feed active
    filename = "{}/image_{:04d}.jpg".format(SAVE_DIR, counterNumber)  # Generate filename
    img.save(filename, quality=90)  # Save image with quality 90
    print("Saved:", filename)

    counterNumber += 1  # Increment image counter
    time.sleep_ms(INTERVAL_MS)  # Wait 3 seconds before the next capture

print("Image capture complete!")
