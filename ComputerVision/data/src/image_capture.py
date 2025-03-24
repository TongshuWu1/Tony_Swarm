import sensor, image, time, os


def openmv_set_saturation_brightness_contrast(saturation: int=0, brightness: int=0, contrast: int=0, ev: int=0):
    # color settings - contrast, brightness, and saturation
    # Do refer to page 49 of this document
    # https://www.arducam.com/downloads/modules/OV5640/OV5640_Software_app_note_parallel.pdf

    # contrast
    sensor.__write_reg(0x3212, 0x03)
    if contrast == 3:
        sensor.__write_reg(0x5586, 0x2c)
        sensor.__write_reg(0x5585, 0x1c)
    elif contrast == 2:
        sensor.__write_reg(0x5586, 0x28)
        sensor.__write_reg(0x5585, 0x18)
    elif contrast == 1:
        sensor.__write_reg(0x5586, 0x24)
        sensor.__write_reg(0x5585, 0x10)
    elif contrast == 0:
        sensor.__write_reg(0x5586, 0x20)
        sensor.__write_reg(0x5585, 0x00)
    elif contrast == -1:
        sensor.__write_reg(0x5586, 0x1c)
        sensor.__write_reg(0x5585, 0x1c)
    elif contrast == -2:
        sensor.__write_reg(0x5586, 0x18)
        sensor.__write_reg(0x5585, 0x18)
    elif contrast == -3:
        sensor.__write_reg(0x5586, 0x14)
        sensor.__write_reg(0x5585, 0x14)

    # brightness
    if brightness == 4:
        sensor.__write_reg(0x5587, 0x40)
        sensor.__write_reg(0x5588, 0x01)
    elif brightness == 3:
        sensor.__write_reg(0x5587, 0x30)
        sensor.__write_reg(0x5588, 0x01)
    elif brightness == 2:
        sensor.__write_reg(0x5587, 0x20)
        sensor.__write_reg(0x5588, 0x01)
    elif brightness == 1:
        sensor.__write_reg(0x5587, 0x10)
        sensor.__write_reg(0x5588, 0x01)
    elif brightness == 0:
        sensor.__write_reg(0x5587, 0x00)
        sensor.__write_reg(0x5588, 0x01)
    elif brightness == -1:
        sensor.__write_reg(0x5587, 0x10)
        sensor.__write_reg(0x5588, 0x09)
    elif brightness == -2:
        sensor.__write_reg(0x5587, 0x20)
        sensor.__write_reg(0x5588, 0x09)
    elif brightness == -3:
        sensor.__write_reg(0x5587, 0x30)
        sensor.__write_reg(0x5588, 0x09)
    elif brightness == -4:
        sensor.__write_reg(0x5587, 0x40)
        sensor.__write_reg(0x5588, 0x09)

    # saturation
    sensor.__write_reg(0x5381, 0x1c)
    sensor.__write_reg(0x5382, 0x5a)
    sensor.__write_reg(0x5383, 0x06)
    if saturation == 3:
        sensor.__write_reg(0x5384, 0x2b)
        sensor.__write_reg(0x5385, 0xab)
        sensor.__write_reg(0x5386, 0xd6)
        sensor.__write_reg(0x5387, 0xda)
        sensor.__write_reg(0x5388, 0xd6)
        sensor.__write_reg(0x5389, 0x04)
    elif saturation == 2:
        sensor.__write_reg(0x5384, 0x24)
        sensor.__write_reg(0x5385, 0x8f)
        sensor.__write_reg(0x5386, 0xb3)
        sensor.__write_reg(0x5387, 0xb6)
        sensor.__write_reg(0x5388, 0xb3)
        sensor.__write_reg(0x5389, 0x03)
    elif saturation == 1:
        sensor.__write_reg(0x5384, 0x1f)
        sensor.__write_reg(0x5385, 0x7a)
        sensor.__write_reg(0x5386, 0x9a)
        sensor.__write_reg(0x5387, 0x9c)
        sensor.__write_reg(0x5388, 0x9a)
        sensor.__write_reg(0x5389, 0x02)
    elif saturation == 0:
        sensor.__write_reg(0x5384, 0x1a)
        sensor.__write_reg(0x5385, 0x66)
        sensor.__write_reg(0x5386, 0x80)
        sensor.__write_reg(0x5387, 0x82)
        sensor.__write_reg(0x5388, 0x80)
        sensor.__write_reg(0x5389, 0x02)
    elif saturation == -1:
        sensor.__write_reg(0x5384, 0x15)
        sensor.__write_reg(0x5385, 0x52)
        sensor.__write_reg(0x5386, 0x66)
        sensor.__write_reg(0x5387, 0x68)
        sensor.__write_reg(0x5388, 0x66)
        sensor.__write_reg(0x5389, 0x02)
    elif saturation == -2:
        sensor.__write_reg(0x5384, 0x10)
        sensor.__write_reg(0x5385, 0x3d)
        sensor.__write_reg(0x5386, 0x4d)
        sensor.__write_reg(0x5387, 0x4e)
        sensor.__write_reg(0x5388, 0x4d)
        sensor.__write_reg(0x5389, 0x01)
    elif saturation == -3:
        sensor.__write_reg(0x5384, 0x0c)
        sensor.__write_reg(0x5385, 0x30)
        sensor.__write_reg(0x5386, 0x3d)
        sensor.__write_reg(0x5387, 0x3e)
        sensor.__write_reg(0x5388, 0x3d)
        sensor.__write_reg(0x5389, 0x01)

    if ev == 3:
        sensor.__write_reg(0x3a0f, 0x60)
        sensor.__write_reg(0x3a10, 0x58)
        sensor.__write_reg(0x3a11, 0xa0)
        sensor.__write_reg(0x3a1b, 0x60)
        sensor.__write_reg(0x3a1e, 0x58)
        sensor.__write_reg(0x3a1f, 0x20)
    elif ev == 2:
        sensor.__write_reg(0x3a0f, 0x50)
        sensor.__write_reg(0x3a10, 0x48)
        sensor.__write_reg(0x3a11, 0x90)
        sensor.__write_reg(0x3a1b, 0x50)
        sensor.__write_reg(0x3a1e, 0x48)
        sensor.__write_reg(0x3a1f, 0x20)
    elif ev == 1:
        sensor.__write_reg(0x3a0f, 0x40)
        sensor.__write_reg(0x3a10, 0x38)
        sensor.__write_reg(0x3a11, 0x71)
        sensor.__write_reg(0x3a1b, 0x40)
        sensor.__write_reg(0x3a1e, 0x38)
        sensor.__write_reg(0x3a1f, 0x10)
    elif ev == 0:
        sensor.__write_reg(0x3a0f, 0x38)
        sensor.__write_reg(0x3a10, 0x30)
        sensor.__write_reg(0x3a11, 0x61)
        sensor.__write_reg(0x3a1b, 0x38)
        sensor.__write_reg(0x3a1e, 0x30)
        sensor.__write_reg(0x3a1f, 0x10)
    elif ev == -1:
        sensor.__write_reg(0x3a0f, 0x30)
        sensor.__write_reg(0x3a10, 0x28)
        sensor.__write_reg(0x3a11, 0x61)
        sensor.__write_reg(0x3a1b, 0x30)
        sensor.__write_reg(0x3a1e, 0x28)
        sensor.__write_reg(0x3a1f, 0x10)
    elif ev == -2:
        sensor.__write_reg(0x3a0f, 0x20)
        sensor.__write_reg(0x3a10, 0x18)
        sensor.__write_reg(0x3a11, 0x41)
        sensor.__write_reg(0x3a1b, 0x20)
        sensor.__write_reg(0x3a1e, 0x18)
        sensor.__write_reg(0x3a1f, 0x10)
    elif ev == -3:
        sensor.__write_reg(0x3a0f, 0x10)
        sensor.__write_reg(0x3a10, 0x08)
        sensor.__write_reg(0x3a11, 0x10)
        sensor.__write_reg(0x3a1b, 0x08)
        sensor.__write_reg(0x3a1e, 0x20)
        sensor.__write_reg(0x3a1f, 0x10)

    sensor.__write_reg(0x538b, 0x98)
    sensor.__write_reg(0x538a, 0x01)
    sensor.__write_reg(0x3212, 0x13)
    sensor.__write_reg(0x3212, 0xa3)


# Initialize the camera
sensor.reset()
sensor.set_pixformat(sensor.RGB565)  # Color images
sensor.set_framesize(sensor.QVGA)    # 320x240 resolution
sensor.skip_frames(time=2000)        # Allow the camera to adjust

# # ISP setup:
# sensor.__write_reg(0x5000, 0b00100111)  # [7]: lens correction, [5]: raw gamma
#                                         # [2:1]: black/white pixel cancellation
#                                         # [0]: color interpolation
# sensor.__write_reg(0x5001, sensor.__read_reg(0x5001) | 0b10000110)# [7]: SFX, [5]: scaling
# # sensor.__write_reg(0x5001, sensor.__read_reg(0x5001) & 0b11011111)  # [2]: UV average,
#                                                                     # [1]: color matrix
#                                                                     # [0]: AWB

openmv_set_saturation_brightness_contrast(saturation=4, brightness=0, contrast=0, ev=0)

sensor.set_auto_exposure(True)
print(sensor.get_rgb_gain_db())
# sensor.set_auto_whitebal(False, rgb_gain_db=(R_GAIN, G_GAIN, B_GAIN))


# Set save directory inside Nicla Vision's internal storage
SAVE_DIR = "/data"

# Ensure the directory exists
try:
    os.mkdir(SAVE_DIR)
except OSError:
    pass  # Directory already exists

# Image capture parameters
NUM_IMAGES = 320 # Capture 5 images
INTERVAL_MS = 2000  # Capture every 3 seconds
counterNumber = 281  # Start file numbering from 1

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
