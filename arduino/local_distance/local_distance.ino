#include <HardwareSerial.h>

#define TFMINI_RX_PIN 9
#define TFMINI_TX_PIN 8

HardwareSerial tfminiSerial(1);

void setup() {
  Serial.begin(115200);  // USB serial to Python
  tfminiSerial.begin(115200, SERIAL_8N1, TFMINI_RX_PIN, TFMINI_TX_PIN);  // TFmini UART
}

void loop() {
  static uint8_t buf[9];
  static int index = 0;

  while (tfminiSerial.available()) {
    uint8_t byteRead = tfminiSerial.read();

    if (index == 0 && byteRead != 0x59) continue;
    if (index == 1 && byteRead != 0x59) {
      index = 0;
      continue;
    }

    buf[index++] = byteRead;

    if (index == 9) {
      index = 0;
      uint8_t checksum = 0;
      for (int i = 0; i < 8; i++) checksum += buf[i];

      if (checksum == buf[8]) {
        uint16_t distance = buf[2] | (buf[3] << 8);

        // Optional: filter invalid data
        if (distance > 0 && distance < 1200) {
          Serial.write('D');
          Serial.write(distance & 0xFF);
          Serial.write((distance >> 8) & 0xFF);
        }
      }
    }
  }

  delay(10);  // ~100Hz output rate
}
