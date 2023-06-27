#include <TinyMLShield.h>

int bytes_per_frame;

byte data[320*240*2];

void setup() {
  Serial.begin(115200); //baud rate
  
  while (!Serial);
  initializeShield();

  Serial.println("OV767X Camera Capture");
  Serial.println();

  if (!Camera.begin(QVGA, RGB565, 1, OV7675)) {
    Serial.println("Failed to initialize camera!");
    while (1);
  }
  
  bytes_per_frame = Camera.width() * Camera.height() * Camera.bytesPerPixel();
  
}

void loop() {
  bool button_clicked = readShieldButton();
  
  if(button_clicked){
    Serial.println("Reading frame");
    Serial.println();

    Camera.readFrame(data);
    Serial.write(data, bytes_per_frame);
    Serial.println("Image Captured");
  }
}
