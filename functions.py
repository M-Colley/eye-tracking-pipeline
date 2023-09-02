import os


import pandas as pd
import shutil
import gc

import re

from PIL import Image, ImageColor

import cv2
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing


from equilib import Equi2Pers


process_lock = multiprocessing.Lock()


reset = "\033[0m"
red = "\033[91m"
green = "\033[92m"
yellow = "\033[93m"
blue = "\033[94m"
purple = "\033[95m"
cyan = "\033[96m"


dpi = 300  # Set the DPI
width, height = 1920, 1080  # Width and height in pixels

# Convert to inches for figsize
width_in = width / dpi
height_in = height / dpi

plt.figure(figsize=(width_in, height_in), dpi=dpi)



colors = {
        "car interior": np.array([128/255, 0/255, 0/255, 0.6]), # Maroon
        "pedestrian": np.array([128/255, 128/255, 0/255, 0.6]), # Olive
        #"helicopters": np.array([0/255, 128/255, 128/255, 0.6]), # Teal
        "buildings": np.array([0/255, 0/255, 128/255, 0.6]), # Navy
        "vehicle": np.array([245/255, 130/255, 48/255, 0.6]), # Orange
        "sky": np.array([128/255, 128/255, 128/255, 0.6]), # Grey
        "road": np.array([70/255, 240/255, 240/255, 0.6]), # Cyan
        #"wall": np.array([70/255, 240/255, 240/255, 0.6]), # Cyan
        #"tree": np.array([145/255, 30/255, 180/255, 0.6]), # Purple
        #"symbol": np.array([255/255, 215/255, 180/255, 0.6]), # Apricot
        "display": np.array([60/255, 180/255, 75/255, 0.6]), # Green
}


class_colors_trust_rgb = {
    'road': np.array([128, 64, 128]),
    'sidewalk': np.array([244, 35, 232]),
    'building': np.array([70, 70, 70]),
    'wall': np.array([102, 102, 156]),
    'fence': np.array([190, 153, 153]),
    'pole': np.array([153, 153, 153]),
    'traffic light': np.array([250, 170, 30]),
    'traffic sign': np.array([220, 220, 0]),
    'vegetation': np.array([107, 142, 35]),
    'terrain': np.array([152, 251, 152]),
    'sky': np.array([70, 130, 180]),
    'person': np.array([220, 20, 60]),
    'rider': np.array([255, 0, 0]),
    'car': np.array([0, 0, 142]),
    'truck': np.array([0, 0, 70]),
    'bus': np.array([0, 60, 100]),
    'train': np.array([0, 80, 100]),
    'motorcycle': np.array([0, 0, 230]),
    'bicycle': np.array([119, 11, 32])
}


circle_radius = 8



def color_distance(color1, color2):
    return sum((a - b) ** 2 for a, b in zip(color1, color2)) ** 0.5

# like this only for trust
def get_class_for_hex(hex_code, threshold=20):
    #target_rgb = hex_to_rgb(hex_code)
    target_rgb = ImageColor.getcolor(hex_code, "RGB")

    closest_class = "NULL"
    min_distance = float("inf")

    for class_name, class_rgb in class_colors_trust_rgb.items():
        #class_rgb = hex_to_rgb(class_hex)
        distance = color_distance(target_rgb, class_rgb)
        if distance < min_distance and distance < threshold:
            closest_class = class_name
            min_distance = distance

    return closest_class


def get_class_for_rgb(target_rgb, threshold=20):
    closest_class = "NULL"
    min_distance = float("inf")

    for class_name, class_rgb in class_colors_trust_rgb.items():
        #class_rgb = hex_to_rgb(class_hex)
        distance = color_distance(target_rgb, class_rgb)
        if distance < min_distance and distance < threshold:
            closest_class = class_name
            min_distance = distance

    return closest_class


def get_hex_code_at_position(pil_image, position):

    # Assuming position is a tuple (x, y)
    x, y = position
    width, height = pil_image.size
    r, g, b = 0, 0, 0
    
    if 0 <= x < width and 0 <= y < height:
        r, g, b = pil_image.getpixel(position)
    else:
        print("Position is out of bounds in get_hex_code_at_position!")
        print("position: " + str(position))
        print("pil_image.size: " + str(pil_image.size))
        print("Setting RGB to 0,0,0")
        
    
    # Convert to hex code
    hex_code = "#{:02x}{:02x}{:02x}".format(r, g, b)

    return hex_code


def get_rgb_at_position(pil_image, position):

    # Assuming position is a tuple (x, y)
    x, y = position
    width, height = pil_image.size
    r, g, b = 0, 0, 0
    
    if 0 <= x < width and 0 <= y < height:
        r, g, b = pil_image.getpixel(position)
    else:
        print("Position is out of bounds in get_hex_code_at_position!")
        print("position: " + str(position))
        print("pil_image.size: " + str(pil_image.size))
        print("Setting RGB to 0,0,0")
        
    
    # Convert to hex code
    hex_code = "#{:02x}{:02x}{:02x}".format(r, g, b)

    return hex_code



# example: color = get_color_for_class("birds")
# Attention: has to be adapted per use case
def get_color_for_class(class_name):
    # from https://sashamaps.net/docs/resources/20-colors/
    # for trust-calibration: pedestrian . vehicle .  buildings . sky . display . road . symbol . roadside


    # If the class name is found in the colors dictionary, return that color
    if class_name in colors:
        return colors[class_name]

    # Otherwise, return a default color for other generic terms
    return np.array([220/255, 190/255, 255/255, 0.6]) # Lavendel


def scale_to_fullHD(original_x, original_y, original_max_x, original_max_y):
    # Check if either value is negative
    if original_x < 0 or original_y < 0:
        return None, None
    
    # out of bounds
    if original_x > original_max_x or original_y > original_max_y:
        return None, None

    # we want to scale to FullHD
    target_max_x = 1920
    target_max_y = 1080

    scaled_x = (original_x / original_max_x) * target_max_x
    scaled_y = (original_y / original_max_y) * target_max_y


    return scaled_x, scaled_y

  


def calculate_view(frame, yaw, pitch):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    equi_img = np.asarray(frame)
    equi_img = np.transpose(equi_img, (2, 0, 1))

    # rotations
    rots = {
        'roll': 0.,
        'pitch': np.deg2rad(pitch),
        'yaw': np.deg2rad(yaw),
    }

    # Intialize equi2pers
    equi2pers = Equi2Pers(
        height=1080,
        width=1920,
        fov_x=110,
        mode="bilinear",
    )

    # obtain perspective image
    pers_img = equi2pers(
        equi=equi_img,
        rots=rots,
    )

    # Transpose the image back to (height, width, channels)
    pers_img = np.transpose(pers_img, (1, 2, 0))

    # Convert to PIL Image
    pers_img_pil = Image.fromarray(pers_img.astype(np.uint8)) 
    
    return pers_img_pil


def process_frame_pre_segmented_360(outputDirs, frame_counter, frame, yaw, pitch, current_eye_gaze, save_image):
    pers_img_pil = calculate_view(frame, yaw, pitch)

    #print(pers_img_pil.size)
    #print(current_eye_gaze)
    # default value
    class_name = "NULL"

    if save_image == True:
        # Add a white filled circle at a specific (x, y) position
        plt.imshow(pers_img_pil)
        plt.axis('off') # To turn off axes
        if current_eye_gaze is not None and current_eye_gaze[0] is not None and current_eye_gaze[1] is not None:
            circle_x = current_eye_gaze[0]  
            circle_y = current_eye_gaze[1]  
            
            plt.scatter(circle_x, circle_y, color='white', s=circle_radius**2, alpha=0.7, edgecolors='none')
            plt.scatter(circle_x, circle_y, color='black', s=circle_radius**2, alpha=0.7, edgecolors='none', marker="+")

        output = os.path.join("EyeGazeOutput",outputDirs[0], outputDirs[1], outputDirs[2])
        if not os.path.exists(output):
            os.makedirs(output)
        plt.savefig(os.path.join(output, "testBild_" + str(frame_counter) +".png"), bbox_inches='tight', pad_inches=0)
        # pers_img_pil.save(os.path.join(output, "testBild_" + str(frame_counter) +".png"))
        plt.clf()
        #print("image saved")

    if current_eye_gaze is not None and current_eye_gaze[0] is not None and current_eye_gaze[1] is not None:
        hex_code = get_hex_code_at_position(pers_img_pil, current_eye_gaze)
        # You can then pass the hex_code to the get_class_for_hex function
        class_name = get_class_for_hex(hex_code)

    return class_name



def process_frame_pre_segmented(iteration, total_participants, id, outputDirs, frame_counter, frame, current_eye_gaze, save_image):
    # apparently was in BGR
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Convert to PIL Image
    pers_img_pil = Image.fromarray(frame.astype(np.uint8)) 

    # default value
    class_name = "NULL"

    if save_image == True:
        # Add a white filled circle at a specific (x, y) position
        plt.imshow(pers_img_pil)
        plt.axis('off') # To turn off axes
        if current_eye_gaze is not None and current_eye_gaze[0] is not None and current_eye_gaze[1] is not None:
            circle_x = current_eye_gaze[0]  
            circle_y = current_eye_gaze[1]  
            
            plt.scatter(circle_x, circle_y, color='white', s=circle_radius**2, alpha=0.7, edgecolors='none')
            plt.scatter(circle_x, circle_y, color='black', s=circle_radius**2, alpha=0.7, edgecolors='none', marker="+")

        output = os.path.join("EyeGazeOutput",outputDirs[0], outputDirs[1], outputDirs[2])
        if not os.path.exists(output):
            os.makedirs(output)
        plt.savefig(os.path.join(output, "testBild_" + str(frame_counter) + "_" + str(id) +".png"), bbox_inches='tight', pad_inches=0)
        # pers_img_pil.save(os.path.join(output, "testBild_" + str(frame_counter) +".png"))
        plt.clf()
        #print("image saved")

    if current_eye_gaze is not None and current_eye_gaze[0] is not None and current_eye_gaze[1] is not None:
        
            # Assuming position is a tuple (x, y)
        x, y = current_eye_gaze
        width, height = pers_img_pil.size
        r, g, b = 0, 0, 0
        
        if 0 <= x < width and 0 <= y < height:
            r, g, b = pers_img_pil.getpixel(current_eye_gaze)
        else:
            print("Position is out of bounds in get_hex_code_at_position!")
            print("position: " + str(current_eye_gaze))
            print("pil_image.size: " + str(pers_img_pil.size))
            print("Setting RGB to 0,0,0")
        
        target_rgb = (r, g, b)

        class_name = get_class_for_rgb(target_rgb)

    outputDirStudy = outputDirs[0]
    outputDirLevel = outputDirs[1]
    outputDirFactor = outputDirs[2]
    formatted_number = "{:04d}".format(frame_counter)
    print_progress_bar(iteration+1, total_participants, prefix=f"{green}{outputDirStudy}{reset} - {blue}{outputDirLevel} {reset} - {red}{outputDirFactor} {reset}", suffix=f"Complete - ID: {id} - frame: {formatted_number}", length=50) 




    return class_name




def process_video(iteration, total_participants, video_path, csv_data, csv_path, current_file_dir_path, prolific_id):
    with process_lock:
        # print(iteration)
        # Create a VideoCapture object
        #print("in process_video")

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

        # Loop through the video and read each frame
        #frames_to_process = []
        frame_counter = 0
        while cap.isOpened():

            ret, frame = cap.read()
            #print(frame_counter)
            if not ret:
                break

            if frame_counter % skip_frames == 0:

                index_to_check = frame_counter // skip_frames
                #print("index_to_check: ", index_to_check)

                # check whether there are still data available
                if index_to_check not in csv_data.index:
                    continue
                if pd.isna(csv_data['displayResX'][index_to_check]):
                    continue

                #yaw = csv_data['yaw'][index_to_check]
                #pitch = csv_data['pitch'][index_to_check]
                gazeX = csv_data['gazeX'][index_to_check]
                gazeY = csv_data['gazeY'][index_to_check]
                displayResX = csv_data['displayResX'][index_to_check]
                displayResY = csv_data['displayResY'][index_to_check] 
                gazeX_Scaled, gazeY_Scaled = scale_to_fullHD(gazeX, gazeY, displayResX, displayResY)

                
                    
                eye_gaze = (gazeX_Scaled, gazeY_Scaled)
                # it is possible that the eye gaze is negative, then this is None, None
                if eye_gaze is not None and all(value is not None and not np.isnan(value) for value in eye_gaze):
                    coords_int = tuple(map(int, np.round(eye_gaze)))  # or np.floor, depends on wishes
                else:
                    coords_int = (None, None)



                # get all relevant data for the current scenario
                normalized_path = os.path.normpath(csv_path)

                # Split the path into individual directory components
                dirs = normalized_path.split(os.path.sep)

                # Extract the last three directory names
                last_three_dirs = dirs[-4:]
                print(last_three_dirs)
                outputDirStudy = last_three_dirs[0]
                outputDirLevel = last_three_dirs[1]
                outputDirFactor = last_three_dirs[2]
                outputdirID = last_three_dirs[3].split("_")[0]
                
                # good for debugging
                #print("Current Video:", last_three_dirs, frame_counter)

                # transform view and find class based on color
                detected_class = process_frame_pre_segmented(iteration, total_participants, prolific_id, last_three_dirs, frame_counter, frame, current_eye_gaze=coords_int, save_image=True)

                # add to CSV
                csv_data.at[index_to_check, "gazeX_Scaled"] = gazeX_Scaled
                csv_data.at[index_to_check, "gazeY_Scaled"] = gazeY_Scaled
                csv_data.at[index_to_check, "detected_class"] = detected_class
            
                

            frame_counter += 1
            # TODO delete
            #if frame_counter > 20:
            #    break

        # Release the VideoCapture object
        # print("release")
        cap.release()
        # collect unreferenced files
        gc.collect()
        # print("csv file array", csv_data)

        # Overwrite the original CSV file
        
        csv_data.to_csv(os.path.join(current_file_dir_path,csv_path), index=False, sep = ';')

        #print("saved csv to: ", str(os.path.join(current_file_dir_path,csv_path)))
        # print("save")
        print_progress_bar(iteration+1, total_participants, prefix=f"{green}{outputDirStudy}{reset} - {blue}{outputDirLevel} {reset} - {red}{outputDirFactor} {reset}", suffix="Complete - CSV Saved", length=50) 
        # print(iteration)
        # time.sleep(1)
        # return 1

def print_progress_bar(iteration, total, prefix='', suffix='', length=50, fill='█'):
    percent = ("{:.1f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end='\r',)
    # if iteration == total:
    #     print()


def count_csv_files(directory_path):
    csv_count = 0
    for filename in os.listdir(directory_path):
        if filename.endswith(".csv"):
            csv_count += 1
    return csv_count

def get_terminal_width():
    return shutil.get_terminal_size().columns

# Function to perform a case-insensitive search for a directory or file
# usually, in Windows these are already case-ins
def find_case_insensitive_path(root_path, target_name):
    for item_name in os.listdir(root_path):
        if re.match(target_name, item_name, re.I):  # re.I flag makes the match case-insensitive
            return os.path.join(root_path, item_name)
    return None
