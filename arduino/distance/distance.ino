#include <WiFi.h>
#include <esp_now.h>
#include <HardwareSerial.h>

#define TFMINI_RX_PIN 9
#define TFMINI_TX_PIN 8

HardwareSerial tfminiSerial(1);

uint8_t receiverMAC[] = {0xDC, 0xDA, 0x0C, 0x57, 0xA7, 0xB8};

QueueHandle_t distanceQueue;

float alpha = 0.95;
float smoothedDistance = 0;
bool isFirst = true;

void tfminiTask(void *parameter) {
  static uint8_t buf[9];
  static int index = 0;

  while (true) {
    if (tfminiSerial.available()) {
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
          xQueueSend(distanceQueue, &distance, portMAX_DELAY);
        }
      }
    }
    vTaskDelay(1);
  }
}

void sendTask(void *parameter) {
  uint16_t distance;

  while (true) {
    if (xQueueReceive(distanceQueue, &distance, portMAX_DELAY) == pdTRUE) {
      if (isFirst) {
        smoothedDistance = distance;
        isFirst = false;
      } else {
        smoothedDistance = alpha * distance + (1 - alpha) * smoothedDistance;
      }

      uint16_t distanceVal = (uint16_t)smoothedDistance;
      uint8_t payload[3] = { 'D', distanceVal & 0xFF, (distanceVal >> 8) & 0xFF };

      esp_err_t result = esp_now_send(receiverMAC, payload, 3);

      // if (result == ESP_OK) {
      //   Serial.print("Sent Distance: ");
      //   Serial.println(distanceVal);
      // } else {
      //   Serial.println("Send failed");
      // }

      vTaskDelay(pdMS_TO_TICKS(100));  // Send every 200 ms
    }
  }
}

void setup() {
  Serial.begin(115200);
  tfminiSerial.begin(115200, SERIAL_8N1, TFMINI_RX_PIN, TFMINI_TX_PIN);

  WiFi.mode(WIFI_STA);
  esp_now_init();

  esp_now_peer_info_t peerInfo = {};
  memcpy(peerInfo.peer_addr, receiverMAC, 6);
  peerInfo.channel = 0;
  peerInfo.encrypt = false;
  esp_now_add_peer(&peerInfo);

  distanceQueue = xQueueCreate(10, sizeof(uint16_t));

  xTaskCreatePinnedToCore(tfminiTask, "TFmini Task", 4096, NULL, 1, NULL, 0);
  xTaskCreatePinnedToCore(sendTask, "Send Task", 4096, NULL, 1, NULL, 1);

  Serial.println("Distance sender ready.");
}

void loop() {
  // Nothing needed
}
