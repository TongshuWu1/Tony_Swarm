import serial
import struct
import time

ser = serial.Serial('/dev/cu.usbmodem11201', 115200, timeout=1)

print("Listening for pressure & distance...")

distance = None
pressure = None

try:
    while True:
        raw = ser.read(3)
        if len(raw) != 3:
            continue

        dtype = chr(raw[0])
        value = struct.unpack('<H', raw[1:])[0]
        timestamp = time.strftime('%H:%M:%S', time.localtime())

        if dtype == 'D':
            distance = value
        elif dtype == 'P':
            pressure = value

        if distance is not None and pressure is not None:
            print(f"[{timestamp}] Distance: {distance} cm | Pressure: {pressure} g")

except KeyboardInterrupt:
    print("Stopped.")
finally:
    ser.close()
