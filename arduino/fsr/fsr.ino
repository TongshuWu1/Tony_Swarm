#include <WiFi.h>
#include <esp_now.h>

#define PRESSURE_PIN A0

uint8_t receiverMAC[] = {0xDC, 0xDA, 0x0C, 0x57, 0xA7, 0xB8};

float alpha = 0.95;
float smoothedPressure = 0;
bool isFirst = true;

void setup() {
  Serial.begin(115200);
  WiFi.mode(WIFI_STA);

  if (esp_now_init() != ESP_OK) {
    Serial.println("ESP-NOW init failed!");
    return;
  }

  esp_now_peer_info_t peerInfo = {};
  memcpy(peerInfo.peer_addr, receiverMAC, 6);
  peerInfo.channel = 0;
  peerInfo.encrypt = false;
  esp_now_add_peer(&peerInfo);

  Serial.println("Pressure sender ready.");
}

void loop() {
  int raw = analogRead(PRESSURE_PIN);
  int pressure = map(raw, 0, 4095, 0, 600);

  if (isFirst) {
    smoothedPressure = pressure;
    isFirst = false;
  } else {
    smoothedPressure = alpha * pressure + (1 - alpha) * smoothedPressure;
  }

  uint16_t pressureVal = (uint16_t)smoothedPressure;
  uint8_t payload[3] = { 'P', pressureVal & 0xFF, (pressureVal >> 8) & 0xFF };

  esp_now_send(receiverMAC, payload, 3);

  // Serial.print("Sent Pressure: ");
  // Serial.println(pressureVal);

  delay(200);  // Send every 200 ms
}
