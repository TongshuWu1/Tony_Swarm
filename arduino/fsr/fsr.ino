#include <WiFi.h>
#include <esp_now.h>

#define PRESSURE_PIN A0

// MAC address of receiver ESP32 (replace with your actual receiver MAC)
uint8_t receiverMAC[] = {0xDC, 0xDA, 0x0C, 0x57, 0xA7, 0xB8};

float alpha = 0.4;
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

  if (esp_now_add_peer(&peerInfo) != ESP_OK) {
    Serial.println("Failed to add peer");
    return;
  }

  Serial.println("Pressure sender ready.");
}

void loop() {
  int raw = analogRead(PRESSURE_PIN);
  int pressure = map(raw, 0, 4095, 0, 600); // Replace with real calibration if needed

  if (isFirst) {
    smoothedPressure = pressure;
    isFirst = false;
  } else {
    smoothedPressure = alpha * pressure + (1 - alpha) * smoothedPressure;
  }

  uint16_t pressureVal = (uint16_t)smoothedPressure;

  uint8_t payload[3];
  payload[0] = 'P';
  payload[1] = pressureVal & 0xFF;
  payload[2] = (pressureVal >> 8) & 0xFF;

  esp_now_send(receiverMAC, payload, 3);

  Serial.print("Sent Pressure: ");
  Serial.println(pressureVal);

  delay(100);  // Send every 100ms
}
