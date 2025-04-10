#include <WiFi.h>
#include <esp_now.h>

void OnDataRecv(const uint8_t *mac, const uint8_t *incomingData, int len) {
  if (len != 3) return;

  char type = incomingData[0];
  uint16_t value = incomingData[1] | (incomingData[2] << 8);

  Serial.write(type);
  Serial.write((uint8_t*)&value, 2);
}

void setup() {
  Serial.begin(115200);
  WiFi.mode(WIFI_STA);
  esp_now_init();
  esp_now_register_recv_cb(OnDataRecv);
}

void loop() {
  // Nothing needed
}
