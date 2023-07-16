# Camera Capture Script

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

