import sensor, image, time

# Initialize the camera
sensor.reset()
sensor.set_pixformat(sensor.GRAYSCALE)  # Grayscale improves edge detection
sensor.set_framesize(sensor.HQVGA)  # Higher resolution for better detection
sensor.skip_frames(time=2000)  # Allow the camera to adjust exposure
sensor.set_gainceiling(8)

clock = time.clock()  # FPS tracking

while True:
    clock.tick()
    img = sensor.snapshot()  # Capture frame

    # Step 1: Apply median filtering to reduce noise while keeping edges sharp
    img.median(1)

    # Step 2: Adaptive thresholding to filter out weak edges
    img.binary([(10, 255)])

    # Step 3: Apply Canny edge detection
    img.find_edges(image.EDGE_CANNY, threshold=(50, 100))

    # Step 4: Close small gaps in edges
    img.close(1)

    # Step 5: Detect blobs (balloons) and filter by shape
    blobs = img.find_blobs([(10, 255)], pixels_threshold=500, area_threshold=500, merge=True)

    for blob in blobs:
        aspect_ratio = blob.w() / blob.h()
        elongation = blob.elongation()  # Closer to 0 = circular, closer to 1 = elongated

        # Accept only rounded blobs (balloons are not perfect circles but are close)
        if 0.8 < aspect_ratio < 1.2 and elongation < 0.5:
            img.draw_rectangle(blob.rect(), color=127)  # Draw bounding box
            img.draw_cross(blob.cx(), blob.cy(), color=127)  # Mark center

    print("Balloons detected:", len(blobs))
    print(clock.fps())  # Display FPS
