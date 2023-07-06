# tinyml-optimization
PS1 Project
IMPORTANT: MAKE PR, DO NOT COMMIT DIRECTLY TO MAIN FOR CHANGES

## Rules
1. Put all .cpp model byte array files in models folder
2. Put all hyperparameters, modifications and accuracy values in comments in the .cpp model file
3. deploy_rps is the folder with deployment files. DO NOT CHANGE ANYTHING HERE

## Instructions
1. Install miniconda: https://docs.conda.io/en/latest/miniconda.html
2. Open Anaconda Prompt (miniconda3)
3. Set up a new conda environment: ```conda create --name tinyml```
4. Enter environment: ```conda activate tinyml```
5. Install required libraries:  ```conda install numpy```,```conda install Pillow```, ```conda install pyserial```
6. Open VSCode inside the environment: ```code .```
7. Run the .ino file on Arduino using the Arduino Web Editor
8. Ensure serial monitor is not open on Arduino Web Editor while trying to run the display_image.py script
9. Run python script
