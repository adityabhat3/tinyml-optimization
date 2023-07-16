# Prepare Model.ipynb

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
