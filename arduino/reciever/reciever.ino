#include <WiFi.h>
#include <esp_now.h>

void OnDataRecv(const uint8_t *mac, const uint8_t *incomingData, int len) {
  if (len != 3) return; // silently ignore invalid packets

  uint8_t low = incomingData[0];
  uint8_t high = incomingData[1];
  uint8_t checksum = incomingData[2];

  if (((low + high) & 0xFF) != checksum) return; // silently ignore bad checksum

  uint16_t distance = low | (high << 8);

  Serial.write((uint8_t*)&distance, 2);  // send raw binary to PC
}

void setup() {
  Serial.begin(115200);
  WiFi.mode(WIFI_STA);
  esp_now_init();
  esp_now_register_recv_cb(OnDataRecv);
}

void loop() {}
