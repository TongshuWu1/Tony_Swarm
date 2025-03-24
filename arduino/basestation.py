import serial
import struct

ser = serial.Serial('/dev/cu.usbmodem111401', 115200, timeout=1)

print("Listening for distance...")

try:
    while True:
        raw = ser.read(2)
        if len(raw) == 2:
            distance = struct.unpack('<H', raw)[0]
            print("Distance:", distance, "cm")
except KeyboardInterrupt:
    print("Stopped.")
finally:
    ser.close()
