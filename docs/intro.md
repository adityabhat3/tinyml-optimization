# Introduction to Project Documentation

This documentation provides an overview and detailed information about the different aspects of the project. It serves as a guide for understanding the project structure, dataset, model files, and other relevant information. Below is the table of contents that includes links to each section for easy navigation:

- [Arduino Deployment](#link1)
- [Final Dataset](Final_Dataset.md)
- [Camera Capture](Camera_Capture.md)
- [Learning Rate Schedulers](Learning_Rate_Schedulers.md)
- [Model Files](Model_Files.md)
- [Prepare Model Notebook](prepare_model.ipynb)

Now let's explore each section in more detail:

# <a name="link1"></a>Arduino Deployment

This Arduino Deployment section provides instructions and guidelines for deploying the project on Arduino-compatible boards. It includes information on prerequisites, installation steps, code overview, and usage instructions.

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

[Go to Arduino Deployment](https://github.com/adityabhat3/tinyml-optimization/tree/main/FINAL/arduino_deployment)

## Final Dataset

The Final Dataset section focuses on documenting the "final_dataset" folder. It explains the folder structure, including the subfolders for rock, paper, and scissors images. Details are provided on how the dataset was created and any considerations for using the images in training.

[Go to Final Dataset](Final_Dataset.md)

## Camera Capture

The Camera Capture section discusses the process of capturing images using the camera module. It covers the necessary dependencies, code implementation, and considerations for capturing images for inference.

[Go to Camera Capture](Camera_Capture.md)

## Learning Rate Schedulers

The Learning Rate Schedulers section provides information on the learning rate scheduling techniques used in the project. It explains the different scheduler classes, their implementations, and how to utilize them for optimizing the model's performance during training.

[Go to Learning Rate Schedulers](Learning_Rate_Schedulers.md)

## Model Files

The Model Files section describes the two important files related to the model. It provides an overview of the "model.tflite" file, explaining its purpose and how to use it for inference. Additionally, it covers the "model64v1.cpp" file, detailing its significance in the deployment process.

[Go to Model Files](Model_Files.md)

## Prepare Model Notebook

The Prepare Model Notebook section is a Jupyter Notebook named "prepare_model.ipynb." It contains the code and instructions for preparing the model, including data preprocessing, model architecture, training, and evaluation.

[Go to Prepare Model Notebook](prepare_model.ipynb)

This documentation aims to provide comprehensive information and guidance throughout the project. Use the links above to navigate to the specific sections of interest.
