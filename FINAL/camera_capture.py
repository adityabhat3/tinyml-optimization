import numpy as np
import serial
import sys
import tensorflow as tf
import uuid
from PIL import Image

# Do not keep serial monitor open on Arduino Web Editor while trying to run this script

port = "/dev/cu.usbmodem14201"  # change based on what is seen in Arduino Web Editor
baudrate = 115200  # do not change baudrate
label = "test"
TFLITE_FILE_PATH = "model.tflite"
NEW_WIDTH = 64
NEW_HEIGHT = 64

# Initialize serial port
ser = serial.Serial()
ser.port = port
ser.baudrate = baudrate
ser.open()
ser.reset_input_buffer()



width = 1
height = 1
num_ch = 3

image = np.empty((height, width, num_ch), dtype=np.uint8)


def serial_readline():
    data = ser.readline()  # read a '\n' terminated line
    return data.decode("utf-8").strip()


interpreter = tf.lite.Interpreter(
    TFLITE_FILE_PATH,
    experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_REF,
)
interpreter.allocate_tensors()

# Get input/output layer information
i_details = interpreter.get_input_details()[0]
o_details = interpreter.get_output_details()[0]

# Get input quantization parameters.
i_quant = i_details["quantization_parameters"]
i_scale = i_quant["scales"][0]
i_zero_point = i_quant["zero_points"][0]


print("Ready")
while True:
    data_str = serial_readline()

    if str(data_str) == "<image>":
        w_str = serial_readline()
        h_str = serial_readline()
        w = int(w_str)
        h = int(h_str)
        if w != width or h != height:
            print("Resizing numpy array")
            if w * h != width * height:
                image.resize((h, w, num_ch))
            else:
                image.reshape((h, w, num_ch))
        width = w
        height = h
        print("Reading frame:", width, height)
        for y in range(0, height):
            for x in range(0, width):
                for c in range(0, num_ch):
                    data_str = serial_readline()
                    image[y][x][c] = int(data_str)

        data_str = serial_readline()
        if str(data_str) == "</image>":
            print("Captured frame")
            resized_image = np.array(image, dtype=np.int8).reshape((-1, NEW_WIDTH, NEW_HEIGHT, 3))
            i_value = resized_image
            i_value = tf.cast(i_value, dtype=tf.int8)

            interpreter.set_tensor(i_details["index"], i_value)
            interpreter.invoke()
            o_pred = interpreter.get_tensor(o_details["index"])[0]
            
            print("Paper Rock Scissors: ")
            print(o_pred)

    else:
        print(data_str)
