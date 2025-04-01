#include <WiFi.h>
#include <esp_now.h>
#include <HardwareSerial.h>

#define TFMINI_RX_PIN 9
#define TFMINI_TX_PIN 8

HardwareSerial tfminiSerial(1);

// MAC address of receiver ESP32
uint8_t receiverMAC[] = {0xDC, 0xDA, 0x0C, 0x57, 0xA7, 0xB8};

void setup() {
  Serial.begin(115200);
  tfminiSerial.begin(115200, SERIAL_8N1, TFMINI_RX_PIN, TFMINI_TX_PIN);

  WiFi.mode(WIFI_STA);
  if (esp_now_init() != ESP_OK) {
    Serial.println("ESP-NOW init failed!");
    return;
  }

  esp_now_peer_info_t peerInfo = {};
  memcpy(peerInfo.peer_addr, receiverMAC, 6);
  peerInfo.channel = 0;
  peerInfo.encrypt = false;

  if (esp_now_add_peer(&peerInfo) != ESP_OK) {
    Serial.println("Failed to add peer");
    return;
  }

  Serial.println("Distance sender ready.");
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

        uint8_t payload[3];
        payload[0] = 'D';
        payload[1] = distance & 0xFF;
        payload[2] = (distance >> 8) & 0xFF;

        esp_err_t result = esp_now_send(receiverMAC, payload, 3);

        if (result == ESP_OK) {
          Serial.print("Sent Distance: ");
          Serial.println(distance);
        } else {
          Serial.println("Send failed");
        }
      }
    }
  }
}
