This documentation provides an overview and detailed information about the different aspects of the project. It serves as a guide for understanding the project structure, dataset, model files, and other relevant information. Below is the table of contents that includes links to each section for easy navigation:

- [Arduino Deployment](#link1)
- [Final Dataset](#link2)
- [Camera Capture](#link3)
- [Model Files](#link4)
- [Prepare Model Notebook](#link5)
- [Learning Rate Schedulers](#link6)

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


# <a name="link2"></a>Final Dataset

The Final Dataset contains subfolders of Rock, Paper and Scissors images used for training the model. The dataset was collecting using the `camera_capture.py` script and setting up the arduino to capture and send raw images via the serial port.

[Go to Final Dataset](https://github.com/adityabhat3/tinyml-optimization/tree/main/FINAL/final_dataset)


# <a name="link3"></a>Camera Capture

This script allows you to capture images from an Arduino device with an attached camera module and perform live inferences using a trained model. It communicates with the Arduino device through a serial connection and receives raw image data.

## Prerequisites

Before running this script, ensure that you have the following:

- Arduino Nano 33 BLE Sense or compatible device
- Camera module (e.g., ov7675) attached to the Arduino
- Python 3.x installed on your system
- Required Python packages installed (numpy, serial, tensorflow, PIL)

## Usage

1. Connect your Arduino device with the camera module to your computer.
2. Set the appropriate port and baudrate variables in the script based on your system configuration.
3. Make sure to have the trained model in TFLite format (`model.tflite`) available.
4. Run the script using the following command: `python camera_capture.py`

## Functionality

1. Capture and Process Image:
   - The script communicates with the Arduino device to receive raw image data.
   - It captures the image, crops it to a square shape, and displays it.
   - You have the option to save the image locally with a provided label.

2. Perform Inference with the Trained Model:
   - The script resizes the captured image to 64x64 pixels.
   - It normalizes the image data and prepares it for inference.
   - The trained model (`model.tflite`) is loaded and used to perform inference on the image.
   - The predicted probabilities for the classes "Paper," "Rock," and "Scissors" are displayed.

## Notes

- The script uses the Python packages numpy, serial, tensorflow, and PIL. Make sure these packages are installed before running the script.
- It assumes that the camera module is properly connected to the Arduino and set up for image capture.
- Adjust the `port` variable in the script to match the correct serial port on your system.
- If needed, modify the `label` and `TFLITE_FILE_PATH` variables in the script according to your requirements.

[Go to Camera Capture](https://github.com/adityabhat3/tinyml-optimization/blob/main/FINAL/camera_capture.py)


# <a name="link4"></a>Model Files

The "model.tflite" file is a TensorFlow Lite model file that contains the trained model for rock-paper-scissors image classification. It is a compact and optimized representation of the model that can be deployed on various platforms.

[Go to TFLite File](https://github.com/adityabhat3/tinyml-optimization/blob/main/FINAL/model.tflite)

The "model64v1.cpp" file is a C++ source code file that is used for deploying the TensorFlow Lite model on an Arduino-compatible board. This file must be copied into `rps_model_dat.cpp` before running the `arduino_deployment.ino` file.

[Go to C source File](https://github.com/adityabhat3/tinyml-optimization/blob/main/FINAL/model64v1.cpp)


# <a name="link5"></a>Preparing The Model 

This Jupyter Notebook provides a step-by-step guide for preparing and training a model using transfer learning on the provided dataset. It utilizes TensorFlow 2.x and the MobileNetV2 architecture for feature extraction.

## Prerequisites

Before running this notebook, ensure that you have the following:

- TensorFlow 2.x installed
- Required Python packages installed (numpy, tensorflow, matplotlib)
- The dataset directory (`final_dataset`) containing the training and validation images

## Notebook Overview

The `prepare_model.ipynb` notebook performs the following steps:

1. Data Loading and Preprocessing:
   - The notebook loads the training and validation datasets using the `tf.keras.utils.image_dataset_from_directory` function.
   - The images are resized to the specified input width and height (`MODEL_INPUT_WIDTH` and `MODEL_INPUT_HEIGHT`).
   - The datasets are normalized using the `tf.keras.layers.Rescaling` layer.

2. Model Architecture:
   - The MobileNetV2 architecture is used as a base model for feature extraction.
   - The top layers of the base model are removed, and additional layers are added for classification.
   - The model is compiled with the Adam optimizer, categorical cross-entropy loss, and accuracy metrics.

3. Data Augmentation:
   - Data augmentation is applied to the training dataset using the `tf.keras.Sequential` and `tf.keras.layers.experimental.preprocessing` modules.
   - Augmentation techniques such as random rotation, horizontal flip, zoom, and translation are used to enhance the dataset.

4. Model Training:
   - The model is trained using the `model.fit` function.
   - The training and validation accuracy and loss are plotted using Matplotlib.

5. Model Evaluation:
   - The trained model is evaluated on the validation dataset.
   - The accuracy and confusion matrix are computed and displayed.

6. Model Conversion to TFLite:
   - The trained model is converted to the TFLite format using the `tf.lite.TFLiteConverter` class.
   - Quantization is applied to the model to optimize size and performance.
   - The TFLite model is saved as `model.tflite`.

7. Model Quantization Parameters:
   - The quantization parameters used in the TFLite model are displayed, including scale and zero-point values.

8. Model Validation with TFLite Interpreter:
   - The TFLite model is loaded using the `tf.lite.Interpreter`.
   - The model is evaluated on the validation dataset using the TFLite interpreter.
   - The accuracy and confusion matrix are computed and displayed.

## Usage

1. Set up the prerequisites and ensure that the dataset is available.
2. Open and run the `prepare_model.ipynb` notebook in a Jupyter Notebook environment or Google Colab.
3. Follow the step-by-step instructions provided in the notebook to train the model, evaluate its performance, and convert it to the TFLite format.
4. Use the generated `model.tflite` file for deployment on resource-constrained platforms.

[Go to Preparing Model Notebook](https://github.com/adityabhat3/tinyml-optimization/blob/main/FINAL/prepare_model.ipynb)


# <a name="link6"></a>Learning Rate Schedulers  

This module provides learning rate schedulers that can be used to adjust the learning rate during the training process. Learning rate scheduling is a technique used in deep learning to control the learning rate based on specific criteria such as epochs or predefined schedules.

To use these learning rate schedulers, you can import the module into your project and instantiate the desired scheduler class. You can then use the scheduler as a callback in the training process to adjust the learning rate over epochs.

## Example usage

from learning_rate_schedulers import StepDecay

lr_callback = tf.keras.callbacks.LearningRateScheduler(StepDecay(initAlpha=1e-3, dropEvery=20, factor=0.75))
model.compile(
  optimizer=tf.keras.optimizers.Adam(),
  loss=tf.losses.SparseCategoricalCrossentropy(from_logits=False),
  metrics=['accuracy']
)

history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=100,
  callbacks=[lr_callback]
)

[Go to Learning Rate Schedulers](https://github.com/adityabhat3/tinyml-optimization/blob/main/FINAL/learning_rate_schedulers.py)

