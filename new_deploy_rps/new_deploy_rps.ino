#include "rps_model_data.cpp"
#include "Arduino.h"
#include <TensorFlowLite.h>
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"

static int bytes_per_frame;
static int bytes_per_pixel;
static bool debug_application = false;

static uint8_t data[160 * 120 * 2]; // QQVGA: 160x120 X 2 bytes per pixel (YUV422)

static int w0 = 0;
static int h0 = 0;
static int stride_in_y = 0;
static int w1 = 0;
static int h1 = 0;
static float scale_x = 0.0f;
static float scale_y = 0.0f;

constexpr int kNumCols = 60;
constexpr int kNumRows = 60;
constexpr int kNumChannels = 3;

constexpr int kMaxImageSize = kNumCols * kNumRows * kNumChannels;

constexpr int kCategoryCount = 3;
constexpr int kRockIndex = 1;
constexpr int kPaperIndex = 0;
constexpr int kScissorsIndex =2;


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
  return (x * scale) - offset;
}

inline int8_t quantize(float x, float scale, float zero_point) {
  return (x / scale) + zero_point;
}

// TensorFlow Lite for Microcontroller global variables
tflite::ErrorReporter* error_reporter = nullptr;
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
static bool g_is_camera_initialized = false;

static constexpr int kTensorArenaSize = 160000;
static uint8_t tensor_arena[kTensorArenaSize];

static uint8_t *tensor_arena = nullptr;
static float   tflu_scale     = 0.0f;
static int32_t tflu_zeropoint = 0;

void tflu_initialization() {
  Serial.println("TFLu initialization - start");

  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;

  // Map the model into a usable data structure. This doesn't involve any
  // copying or parsing, it's a very lightweight operation.
  model = tflite::GetModel(g_rps_model_data);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    TF_LITE_REPORT_ERROR(error_reporter,
                         "Model provided is schema version %d not equal "
                         "to supported version %d.",
                         model->version(), TFLITE_SCHEMA_VERSION);
    return;
  }

  static tflite::MicroMutableOpResolver<8> micro_op_resolver;
  micro_op_resolver.AddMean();
  micro_op_resolver.AddAdd();
  micro_op_resolver.AddDequantize();
  micro_op_resolver.AddConv2D();
  micro_op_resolver.AddDepthwiseConv2D();
  micro_op_resolver.AddFullyConnected();
  micro_op_resolver.AddPad();
  micro_op_resolver.AddSoftmax();

  // Initialize the TFLu interpreter
  static tflite::MicroInterpreter static_interpreter(
      model, micro_op_resolver, tensor_arena, kTensorArenaSize, error_reporter);
  interpreter = &static_interpreter;

  input = interpreter->input(0);

  // Allocate TFLu internal memory
  tflu_interpreter->AllocateTensors();

  const auto* i_quantization = reinterpret_cast<TfLiteAffineQuantization*>(input->quantization.params);

  // Get the quantization parameters (per-tensor quantization)
  float tflu_scale     = i_quantization->scale->data[0];
  int32_t tflu_zeropoint = i_quantization->zero_point->data[0];

  // Allocate memory from the tensor_arena for the model's tensors.
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    TF_LITE_REPORT_ERROR(error_reporter, "AllocateTensors() failed");
    return;
  }
  Serial.println("TFLu initialization - completed");
}

void setup() {
  Serial.begin(115600);
  while (!Serial);

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
  //uint8_t rgb888[3];
  bytes_per_pixel = Camera.bytesPerPixel();
  bytes_per_frame = Camera.width() * Camera.height() * bytes_per_pixel;
  // Initialize resolution
  w0 = Camera.height();
  h0 = Camera.height();
  stride_in_y = Camera.width() * bytes_per_pixel;
  w1 = 60;
  h1 = 60;

  // Initialize scaling factors
  scale_x = (float)w0 / (float)w1;
  scale_y = (float)h0 / (float)h1;
  // Initialize TFLu
  tflu_initialization();
}

void loop() {
  Camera.readFrame(data);
  uint8_t rgb888[3];
  if(debug_application) {
    Serial.println("<image>");
    Serial.println(w1);
    Serial.println(h1);
  }

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
        c_f = rescale((float)c_i, 1.f/127.5f, -1.f);
        c_q = quantize(c_f, tflu_scale, tflu_zeropoint);
        input->data.int8[idx++] = c_q;
        if(debug_application) {
          Serial.println(c_i);
        }
      }
    }
  }
  if(debug_application) {
    Serial.println("</image>");
  }
  // Run inference
  if (kTfLiteOk != interpreter->Invoke()) {
    TF_LITE_REPORT_ERROR(error_reporter, "Invoke failed.");
  }

  TfLiteTensor* output = interpreter->output(0);

  // Process the inference results.
  float rock_score = output->data.f[kRockIndex];
  float paper_score = output->data.f[kPaperIndex];
  float scissors_score = output->data.f[kScissorsIndex];

  static bool is_initialized = false;
  if (!is_initialized) {
    // Pins for the built-in RGB LEDs on the Arduino Nano 33 BLE Sense
    pinMode(LEDR, OUTPUT); //rock
    pinMode(LEDG, OUTPUT); //paper
    pinMode(LEDB, OUTPUT); //scissor
    is_initialized = true;
  }
  float rock_factor=1.0;

  // Note: The RGB LEDs on the Arduino Nano 33 BLE
  // Sense are on when the pin is LOW, off when HIGH.

  // Switch the person/not person LEDs off
  digitalWrite(LEDR, HIGH);
  digitalWrite(LEDG, HIGH);
  digitalWrite(LEDB, HIGH);
  delay(100);

  if (rock_factor*rock_score > paper_score) {
    if(rock_factor*rock_score > scissors_score){
      digitalWrite(LEDR, LOW);
    }
    else{
      digitalWrite(LEDB, LOW);
    }
  } 
  else {
    if(paper_score > scissors_score){
      digitalWrite(LEDG, LOW);
    }
    else{
      digitalWrite(LEDB, LOW);
    }
  }

  int rock_percentage = (int)100*rock_score;
  int paper_percentage = (int)100*paper_score;
  int scissors_percentage = (int)100*scissors_score;



  TF_LITE_REPORT_ERROR(error_reporter, "Rock percentage: %d Paper percentage: %d Scissors percentage: %d",
                       rock_percentage, paper_percentage, scissors_percentage);

}
