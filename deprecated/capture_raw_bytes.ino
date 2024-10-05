#include <TinyMLShield.h>

int bytes_per_frame;

byte data[320*240*2];

void rgb565_rgb888(uint8_t* in, uint8_t* out) {
  uint16_t p = (in[0] << 8) | in[1];
  out[0] = ((p >> 11) & 0x1f) << 3;
  out[1] = ((p >> 5) & 0x3f) << 2;
  out[2] = (p & 0x1f) << 3;
}

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
  
  Camera.testPattern(); //comment to stop test

  
  bytes_per_frame = Camera.width() * Camera.height() * Camera.bytesPerPixel();
  
}

void loop() {
  bool button_clicked = readShieldButton();
  
  if(button_clicked){
    Camera.readFrame(data);
    uint8_t rgb888[3];
    Serial.println("<image>");    
    Serial.println(Camera.width());
    Serial.println(Camera.height());
    const int bytes_per_pixel = Camera.bytesPerPixel();
    
    for(int i = 0; i < bytes_per_frame; i+=bytes_per_pixel) {
      rgb565_rgb888(&data[i], &rgb888[0]);
      Serial.println(rgb888[0]);
      Serial.println(rgb888[1]);
      Serial.println(rgb888[2]);
    }
    Serial.println("</image>");
  }
}
