#!/usr/bin/env/python
import cv2
import glob
import re
import pandas as pd
import os
import numpy as np
from PIL import Image
import torch
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import math

import gc


# from config import *
from functions import *

#os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

print(os.environ['CUDA_HOME'])
print("CUDA is available: " + str(torch.cuda.is_available()))


current_file_dir_path = os.path.dirname(os.path.abspath(__file__))

# Directory paths
video_directory = ".\Videos"
tracking_data_directory = "FOV_Tracking_Data"
circle_radius = 8

# Function to perform a case-insensitive search for a directory or file
# usually, in Windows these are already case-ins
def find_case_insensitive_path(root_path, target_name):
    for item_name in os.listdir(root_path):
        if re.match(target_name, item_name, re.I):  # re.I flag makes the match case-insensitive
            return os.path.join(root_path, item_name)
    return None


# Iterate over all .csv files in the video directory
for video_file in glob.glob(os.path.join(video_directory, "*.mp4")):
    # Extract the csv name without extension
    print("video_path: " + video_file)


    # Create a VideoCapture object
    cap = cv2.VideoCapture(video_file)

    # Check if the video file is opened successfully
    if not cap.isOpened():
        print("Error: Could not open video file.")
        exit()

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Initialize VideoWriter
    output_path = video_file[:-4] + "_segmented.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    print("output_path: " + output_path)

    # Target frame rate (10Hz)
    #target_framerate = 10

    frame_counter = 0
    while cap.isOpened():

        ret, frame = cap.read()
        #print(frame_counter)
        if not ret:
            break

        # not relevant anymore
        text_prompt_custom = "pedestrian . vehicle .  buildings . sky . display . road . car interior"

        #height, width = frame.shape[:2]
        #print("Dimensions: Width = {}, Height = {}".format(width, height))

        detected_class, segmented_image = process_image(image_source=Image.fromarray(frame.astype(np.uint8)), current_eye_gaze=(0,0), text_prompt=text_prompt_custom, return_segmented_image=True, show_boxes=False, return_detected_class=False)

        # GroundingDINO also checks combinations, make sure this is not included in the final .csv
        if "." in detected_class:
            detected_class = "NULL"

        
        segmented_image.axis("off")
        #segmented_image.tight_layout()

        #segmented_image.savefig(".\Test.png", bbox_inches='tight', pad_inches=0)
        
        #print(segmented_image.shape)

        #image_source = cv2.resize(image_source, (1920, 1080))

        canvas = segmented_image.gca().figure.canvas
        canvas.draw()
        img_arr = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
        img_arr = img_arr.reshape(canvas.get_width_height()[::-1] + (3,))

        #img_arr = cv2.resize(img_arr, (1920, 1080))


        # Convert to PIL Image
        pil_image = Image.fromarray(img_arr)
        #pil_image.save(".\Test_PIL.png")

        # Convert PIL image to OpenCV format
        opencv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

        # Before writing frame
        h, w, _ = opencv_image.shape
        if h != height or w != width:
            print(f"Error: Dimension mismatch, expected {height}x{width}, got {h}x{w}")
            exit()

        out.write(opencv_image)
        
        # after saving, delete it
        del segmented_image

        frame_counter += 1
        # TODO delete
        #if frame_counter > 100:
        #    break

    # Release the VideoCapture object
    cap.release()
    out.release()
    # collect unreferenced files
    gc.collect()

    print()
    print("#######################################")
    print("#######################################")
    print("Video done")
    print("#######################################")
    print("#######################################")