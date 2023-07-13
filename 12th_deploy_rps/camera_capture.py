import numpy as np
import serial
import sys
import tensorflow as tf
import uuid
from PIL import Image
#IMPORTANT: Do not keep serial monitor open on Arduino Web Editor while trying to run this script
port = '/dev/cu.usbmodem14201' # change based on what is seen in Arduino Web Editor
baudrate = 115200 # do not change baudrate
label="test"
TFLITE_FILE_PATH=""

model=tf.keras.models.load_model("model60x60v6")
# model.eval()
# Initialize serial port
ser = serial.Serial()
ser.port     = port
ser.baudrate = baudrate
ser.open()
ser.reset_input_buffer()

width  = 1
height = 1
num_ch = 3

image = np.empty((height, width, num_ch), dtype=np.uint8)

def serial_readline():
    data = ser.readline() # read a '\n' terminated line
    return data.decode("utf-8").strip()

interpreter = tf.lite.Interpreter(TFLITE_FILE_PATH)


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
            width  = w
            height = h
        print("Reading frame:", width, height)
        for y in range(0, height):
            for x in range(0, width):
                for c in range(0, num_ch):
                    data_str = serial_readline()
                    image[y][x][c] = int(data_str)
        # print(image.shape)
        data_str = serial_readline()
        if str(data_str) == "</image>":
            print("Captured frame")
            crop_area = (0, 0, height, height)
            image_pil = Image.fromarray(image)
            # print(image_pil.shape())
            image_cropped = image_pil.crop(crop_area)
            image_cropped.show()
            # img=image_cropped.resize((48,48))
            # print(model.predict(np.asarray(img)))
            key = input("Save image? [y] for YES: ")
            if key == 'y':
                str_label = f"Write label or leave it blank to use [{label}]: "
                label_new = input(str_label)
                if label_new != '':
                    label = label_new
                unique_id = str(uuid.uuid4())
                filename = label + "_"+ unique_id + ".png"
                image_cropped.save(filename)
                print(f"Image saved as {filename}\n")

                # from PIL import Image

                new_image = Image.open(f"{filename}")
                new_width = 64
                new_height = 64
            
                resized_image = np.array(new_image.resize((new_width, new_height)), dtype=int).reshape((-1,60,60,3))
                resized_image = resized_image / 127.5 -1 # Assuming your inference data is in the range [0, 255]
                # resized_image = (resized_image - 0.5) * 2.0
                print("Paper Rock Scissors: ")
                print(model.predict(resized_image))

   
        elif str(data_str).startswith("Rock"):
            print(str(data_str)) # Arduino inference

        else:
            print("Error capturing image\n")


        