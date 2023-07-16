# Rock-Paper-Scissors TensorFlow Lite Model

This project demonstrates the implementation of a Rock-Paper-Scissors TensorFlow Lite model on an Arduino-compatible device. The model is used to classify hand gestures of rock, paper, or scissors in real-time.

## Getting Started

### Prerequisites

- Arduino IDE
- Arduino-compatible board (such as Arduino Uno or Arduino Nano)
- Camera module compatible with your board (e.g., OV7670)
- TensorFlow Lite library for Arduino

### Installation

1. Clone or download the project repository.

2. Open the Arduino IDE.

3. Install the TensorFlow Lite library for Arduino by following the instructions provided by the library's repository.

4. Connect your board to your computer.

5. Open the `rps_model.ino` sketch from the project directory in the Arduino IDE.

6. Compile and upload the sketch to your Arduino-compatible board.

7. Ensure that your camera module is connected to the appropriate pins on your board, as defined in the sketch.

8. Run the sketch on your board.

## Code Overview

### Dependencies

- `TensorFlowLite.h`: The TensorFlow Lite library for Arduino.
- `main_functions.h`: Helper functions for capturing images and processing inferences.
- `detection_responder.h`: Functions for responding to the detection results.
- `image_provider.h`: Functions for capturing images from the camera module.
- `model_settings.h`: Configuration settings for the model.
- `rps_model_data.h`: The binary representation of the trained TensorFlow Lite model.

### Global Variables

- `error_reporter`: An error reporter for logging messages.
- `model`: The TensorFlow Lite model loaded from `rps_model_data.h`.
- `interpreter`: The TensorFlow Lite interpreter to run the model.
- `input`: The input tensor of the model.
- `bytes_per_frame`: The number of bytes per frame from the camera module.
- `bytes_per_pixel`: The number of bytes per pixel in the image.
- `data`: An array to store the captured image data.
- `tflu_scale`: The scale value used for input quantization.
- `tflu_zeropoint`: The zero point value used for input quantization.
- `tensor_arena`: The memory allocated for the model's tensors.

### Setup

The `setup` function initializes the necessary components and sets up the error reporter, model, interpreter, and input tensor.

### Loop

The `loop` function is the main program loop. It captures an image from the camera module, runs the model inference, and processes the results.

## Usage

1. Connect the Arduino-compatible board and camera module.

2. Upload the sketch to your board.

3. Observe the serial monitor for the detection results.

4. Show your hand gestures in front of the camera module, and the model will classify them as rock, paper, or scissors.

## Troubleshooting

- If the model fails to load or invoke, ensure that the model schema version matches the supported version defined in `TFLITE_SCHEMA_VERSION`.

- If the image capture fails, check the camera module connection and compatibility with your board.

- If you encounter any other issues, refer to the error messages in the serial monitor for troubleshooting guidance.


