#!/usr/bin/env/python
import cv2
import glob
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
from functions_grounding_dino import *



# ------------------------------
# DEPRECATED FILE
# This file is deprecated and will be removed in future versions.
# Use `new_file.py` instead.
# ------------------------------

import warnings

# Issue a deprecation warning when the file is imported
warnings.warn(
    "This module is deprecated; use main_trust_calibration_parallel_eye_tracking.py instead.",
    DeprecationWarning,
    stacklevel=2
)


#os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

print(os.environ['CUDA_HOME'])
print("CUDA is available: " + str(torch.cuda.is_available()))


current_file_dir_path = os.path.dirname(os.path.abspath(__file__))

# Directory paths
video_directory = ".\Videos"
tracking_data_directory = "FOV_Tracking_Data"
circle_radius = 8


# Iterate through all subdirectories under the "Videos" directory
for study_dir in os.listdir(tracking_data_directory): # Schlecht/Gut
    for video_dir in os.listdir(os.path.join(tracking_data_directory, study_dir)): # Video1-4
        # Full path to the video directory
        video_subdir_path = os.path.join(tracking_data_directory, study_dir, video_dir)
        # Iterate through one more list of subdirectories
        for subdirectory in os.listdir(video_subdir_path):
            subdirectory_path = os.path.join(video_subdir_path, subdirectory)
            # Now you can work with the subdirectory_path as needed

            # Iterate over all .csv files in the video directory
            for csv_file in glob.glob(os.path.join(subdirectory_path, "*.csv")):
                # Extract the csv name without extension
                csv_name = os.path.splitext(os.path.basename(csv_file))[0]
                #print(video_name)
                # Split the video name by "_" and take the last part
                #tracking_subdir_name = csv_name.split("_")[-1]
                #print(tracking_subdir_name)
                #current_file_dir_path = os.path.dirname(os.path.abspath(__file__))

                # get corresponding video
                video_path = os.path.join(video_directory, video_dir + ".mp4")
                csv_path = csv_file

                print("video_path: " + video_path)
                print("csv_path: " + csv_path)

                # Read the CSV file
                csv_data = pd.read_csv(os.path.join(current_file_dir_path,csv_path), delimiter=";")
                # print(csv_data.head())

                # Create a VideoCapture object
                cap = cv2.VideoCapture(video_path)

                # Check if the video file is opened successfully
                if not cap.isOpened():
                    print("Error: Could not open video file.")
                    exit()

                # Get the video's original frame rate
                original_framerate = int(cap.get(cv2.CAP_PROP_FPS))

                # Target frame rate (10Hz)
                target_framerate = 10

                # Calculate the number of frames to skip
                skip_frames = original_framerate // target_framerate

                frame_counter = 0
                while cap.isOpened():

                    ret, frame = cap.read()
                    #print(frame_counter)
                    if not ret:
                        break

                    if frame_counter % skip_frames == 0:

                        index_to_check = frame_counter // skip_frames


                        # check whether there are still data available
                                # check whether there are still data available
                        if index_to_check not in csv_data.index:
                            continue
                        if pd.isna(csv_data['yaw'][index_to_check]):
                            continue
                        
                        gazeX = csv_data['gazeX'][index_to_check]
                        gazeY = csv_data['gazeY'][index_to_check]
                        displayResX = csv_data['displayResX'][index_to_check]
                        displayResY = csv_data['displayResY'][index_to_check] 
                        #1516.4131910790093	-48.71241166285146	2400	1600
                        gazeX_Scaled, gazeY_Scaled = scale_to_fullHD(gazeX, gazeY, displayResX, displayResY)
                        # add to csv
                        csv_data.loc['gazeX_Scaled'][index_to_check] = gazeX_Scaled
                        csv_data.loc['gazeY_Scaled'][index_to_check] = gazeY_Scaled

                        #eye_gaze = tuple([gazeX_Scaled,gazeY_Scaled])
                        #coords_int = np.round(eye_gaze).astype(int)  # or np.floor, depends on wishes   
                                                    
                        print("X: " + str(gazeX_Scaled))
                        print("Y: " + str(gazeY_Scaled))
                        
                        eye_gaze = (gazeX_Scaled, gazeY_Scaled)
                        coords_int = tuple(map(int, np.round(eye_gaze)))  # or np.floor, depends on wishes
                        print(coords_int)
                        # get all relevant data for the current scenario
                        normalized_path = os.path.normpath(csv_path)

                        # Split the path into individual directory components
                        dirs = normalized_path.split(os.path.sep)

                        # Extract the last three directory names
                        last_three_dirs = dirs[-4:]
                        outputDirStudy = last_three_dirs[0]
                        outputDirLevel = last_three_dirs[1]
                        outputDirFactor = last_three_dirs[2]
                        outputdirID = last_three_dirs[3].split("_")[0]
                        # good for debugging
                        print("Last three directory names:", last_three_dirs)

                        # create dir
                        dir_path = os.path.join("SegmentedFrames", outputDirStudy, outputDirLevel, outputDirFactor, outputdirID)
                        
                        if not os.path.exists(dir_path):
                            os.makedirs(dir_path)
                        
                        # not relevant anymore
                        text_prompt_custom = "pedestrian . vehicle .  buildings . sky . display . road . car interior . roadside"

                        #height, width = frame.shape[:2]
                        #print("Dimensions: Width = {}, Height = {}".format(width, height))

                        detected_class, segmented_image = process_image(image_source=Image.fromarray(frame.astype(np.uint8)), current_eye_gaze=coords_int, text_prompt=text_prompt_custom, return_segmented_image=True, show_boxes=False, return_detected_class=False)
                        # add to CSV

                        # GroundingDINO also checks combinations, make sure this is not included in the final .csv
                        if "." in detected_class:
                            detected_class = "NULL"

                        csv_data.loc['detected_class'][index_to_check] = detected_class
                        print("detected_class: " + detected_class)

                        # add gaze point
                        ax = plt.gca()

                        # Get the X and Y axis limits
                        x_limit = math.ceil(ax.get_xlim()[1])
                        y_limit = math.ceil(ax.get_ylim()[0])

                        
                        # Add a red filled circle at a specific (x, y) position
                        circle_x = gazeX_Scaled  
                        circle_y = gazeY_Scaled  
                        
                        plt.scatter(circle_x, circle_y, color='white', s=circle_radius**2, alpha=0.7, edgecolors='none')
                        plt.scatter(circle_x, circle_y, color='black', s=circle_radius**2, alpha=0.7, edgecolors='none', marker="+")

                        # Save to PNG file
                        output_path = os.path.join(dir_path, f'{outputDirStudy}_{outputDirLevel}_{outputDirFactor}_{outputdirID}_frame_{frame_counter}.png')
                        segmented_image.axis("off")
                        
                        if segmented_image is not None:
                            segmented_image.savefig(output_path, bbox_inches='tight', pad_inches=0)
                            #segmented_image.save(output_path)
                        
                        # after saving, delete it
                        del segmented_image

                    frame_counter += 1
                    # TODO delete
                    if frame_counter > 350:
                        break

                # Release the VideoCapture object
                cap.release()
                # collect unreferenced files
                gc.collect()

                # Overwrite the original CSV file
                csv_data.to_csv(os.path.join(current_file_dir_path,csv_path), index=False, sep = ';')


                print()
                print("#######################################")
                print("#######################################")
                print("Video done")
                print("#######################################")
                print("#######################################")