/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.
  Licensed under the Apache License, Version 2.0 (the "License");
  you may not use this file except in compliance with the License.
  You may obtain a copy of the License at
  http://www.apache.org/licenses/LICENSE-2.0
  Unless required by applicable law or agreed to in writing, software
  distributed under the License is distributed on an "AS IS" BASIS,
  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  See the License for the specific language governing permissions and
  limitations under the License.
  ==============================================================================*/

#include "image_provider.h"

#ifndef ARDUINO_EXCLUDE_CODE

#include "Arduino.h"
#include <TinyMLShield.h>

namespace{
static int bytes_per_frame;
static int bytes_per_pixel;
static bool debug_application = true;
static float   tflu_scale     = 0.0f;
static int32_t tflu_zeropoint = 0;
static int w0 = 0;
static int h0 = 0;
static int stride_in_y = 0;
static int w1 = 0;
static int h1 = 0;
static float scale_x = 0.0f;
static float scale_y = 0.0f;
}
template <typename T>
inline T clamp_0_255(T x) {
  return std::max(std::min(x, static_cast<T>(255)), static_cast<T>(0));
}

inline void ycbcr422_rgb888(int32_t Y, int32_t Cb, int32_t Cr, uint8_t* out) {
  Cr = Cr - 128;
  Cb = Cb - 128;

  out[0] = clamp_0_255((int)(Y + Cr + (Cr >> 2) + (Cr >> 3) + (Cr >> 5)));
  out[1] = clamp_0_255((int)(Y - ((Cb >> 2) + (Cb >> 4) + (Cb >> 5)) - ((Cr >> 1) + (Cr >> 3) + (Cr >> 4)) + (Cr >> 5)));
  out[2] = clamp_0_255((int)(Y + Cb + (Cb >> 1) + (Cb >> 2) + (Cb >> 6)));
}

inline uint8_t bilinear_inter(uint8_t v00, uint8_t v01, uint8_t v10, uint8_t v11, float xi_f, float yi_f, int xi, int yi) {
    const float a  = (xi_f - xi);
    const float b  = (1.f - a);
    const float a1 = (yi_f - yi);
    const float b1 = (1.f - a1);

    // Calculate the output
    return clamp_0_255((v00 * b * b1) + (v01 * a * b1) + (v10 * b * a1) + (v11 * a * a1));
}

inline float rescale(float x, float scale, float offset) {
  return (x * scale) + offset;
}

inline int8_t quantize(float x, float scale, float zero_point) {
  return (x / scale) + zero_point;
}

// Get an image from the camera module
TfLiteStatus GetImage(tflite::ErrorReporter* error_reporter, int image_width,
                      int image_height, int channels, int8_t* image_data, float tflu_scale, int32_t tflu_zeropoint) {

  static uint8_t data[160 * 120 * 2]; // QQVGA: 160x120 X 2 bytes per pixel (YUV422)

  static bool g_is_camera_initialized = false;

  // Initialize camera if necessary
  if (!g_is_camera_initialized) {
    if (!Camera.begin(QQVGA, YUV422, 1, OV7675)) {
      TF_LITE_REPORT_ERROR(error_reporter, "Failed to initialize camera!");
      return kTfLiteError;
    }
    g_is_camera_initialized = true;
  }

  // Read camera data
  Camera.readFrame(data);
  uint8_t rgb888[3];
  
  bytes_per_pixel = Camera.bytesPerPixel();
  bytes_per_frame = Camera.width() * Camera.height() * bytes_per_pixel;
  // Initialize resolution
  w0 = Camera.height();
  h0 = Camera.height();
  stride_in_y = Camera.width() * bytes_per_pixel;
  w1 = 64;
  h1 = 64;

  if(debug_application){
    Serial.println("<image>");
    Serial.println(w1);
    Serial.println(h1);
  }

  // Initialize scaling factors
  scale_x = (float)w0 / (float)w1;
  scale_y = (float)h0 / (float)h1;
  

  int idx = 0;
  for (int yo = 0; yo < h1; yo++) {
    const float yi_f = (yo * scale_y);
    const int yi = (int)std::floor(yi_f);
    for(int xo = 0; xo < w1; xo++) {
      const float xi_f = (xo * scale_x);
      const int xi = (int)std::floor(xi_f);

      int x0 = xi;
      int y0 = yi;
      int x1 = std::min(xi + 1, w0 - 1);
      int y1 = std::min(yi + 1, h0 - 1);

      // Calculate the offset to access the Y component
      int ix_y00 = x0 * sizeof(int16_t) + y0 * stride_in_y;
      int ix_y01 = x1 * sizeof(int16_t) + y0 * stride_in_y;
      int ix_y10 = x0 * sizeof(int16_t) + y1 * stride_in_y;
      int ix_y11 = x1 * sizeof(int16_t) + y1 * stride_in_y;

      const int Y00 = data[ix_y00];
      const int Y01 = data[ix_y01];
      const int Y10 = data[ix_y10];
      const int Y11 = data[ix_y11];

      // Calculate the offset to access the Cr component
      const int offset_cr00 = xi % 2 == 0? 1 : -1;
      const int offset_cr01 = (xi + 1) % 2 == 0? 1 : -1;

      const int Cr00 = data[ix_y00 + offset_cr00];
      const int Cr01 = data[ix_y01 + offset_cr01];
      const int Cr10 = data[ix_y10 + offset_cr00];
      const int Cr11 = data[ix_y11 + offset_cr01];

      // Calculate the offset to access the Cb component
      const int offset_cb00 = offset_cr00 + 2;
      const int offset_cb01 = offset_cr01 + 2;

      const int Cb00 = data[ix_y00 + offset_cb00];
      const int Cb01 = data[ix_y01 + offset_cb01];
      const int Cb10 = data[ix_y10 + offset_cb00];
      const int Cb11 = data[ix_y11 + offset_cb01];

      uint8_t rgb00[3];
      uint8_t rgb01[3];
      uint8_t rgb10[3];
      uint8_t rgb11[3];

      // Convert YCbCr422 to RGB888
      ycbcr422_rgb888(Y00, Cb00, Cr00, rgb00);
      ycbcr422_rgb888(Y01, Cb01, Cr01, rgb01);
      ycbcr422_rgb888(Y10, Cb10, Cr10, rgb10);
      ycbcr422_rgb888(Y11, Cb11, Cr11, rgb11);

      // Iterate over the RGB channels
      uint8_t c_i;
      float c_f;
      int8_t c_q;
      for(int i = 0; i < 3; i++) {
        c_i = bilinear_inter(rgb00[i], rgb01[i], rgb10[i], rgb11[i], xi_f, yi_f, xi, yi);
        c_f = rescale((float)c_i, 1.0f/127.5f, -1.0f);
        c_q = quantize(c_f, tflu_scale, tflu_zeropoint);
        image_data[idx++] = c_q;
        if(debug_application){
          Serial.println(c_q);
        }
      }
    }
  }
  if(debug_application){
    Serial.println("</image>");
  }

  return kTfLiteOk;
}

#endif //ARDUINO_EXCLUDE_CODE