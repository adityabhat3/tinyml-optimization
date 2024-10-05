import cv2
import os

folder_path = 'edited_dataset_grayscale/scissors'

for filename in os.listdir(folder_path):
    if filename.endswith(('.jpg', '.jpeg', '.png')):  # Consider only image files
        # Read the image
        image_path = os.path.join(folder_path, filename)
        image = cv2.imread(image_path)

        # Convert the image to grayscale
        grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Save the grayscale image
        grayscale_image_path = os.path.join(folder_path, filename)
        cv2.imwrite(grayscale_image_path, grayscale_image)
        print(filename)

