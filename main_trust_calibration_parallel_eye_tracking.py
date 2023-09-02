#!/usr/bin/env/python
#from alive_progress import alive_bar
import os, glob
#import pandas as pd
import concurrent.futures
import warnings
#from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import multiprocessing

# Filter out Pandas warnings
warnings.filterwarnings("ignore")


# from config import *
from functions import *

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# print(os.environ['CUDA_HOME'])
# print("CUDA is available: " + str(torch.cuda.is_available()))

# Get the number of available CPU cores
num_cores = os.cpu_count() or 1
#num_cores = 1 # for debugging purposes
print("max number of cores:", num_cores)
# print("Threat started")

current_file_dir_path = os.path.dirname(os.path.abspath(__file__))

# Directory paths
video_directory = ".\VideosSegmented"
tracking_data_directory = ".\FOV_Tracking_Data"
process_lock = multiprocessing.Lock()


# Iterate through all subdirectories under the "VideosSegmented" directory
# this is Intro ambiguous or boasting
if __name__ == '__main__':
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_cores) as executor:
        futures = []  # Initialize a list to hold future results
        for intro_dir in os.listdir(video_directory):
            for scenario_condition_dir in os.listdir(os.path.join(video_directory, intro_dir)):
                for condition_dir in os.listdir(os.path.join(video_directory, intro_dir, scenario_condition_dir)):
                    
                    total_participants_in_condition = count_csv_files(os.path.join(tracking_data_directory, intro_dir, scenario_condition_dir, condition_dir))
                    #print("total_participants_in_condition", total_participants_in_condition)

                    for video_file in os.listdir(os.path.join(video_directory, intro_dir, scenario_condition_dir, condition_dir)):
                        video_name = os.path.splitext(os.path.basename(video_file))[0]
                        video_path = os.path.join(video_directory, intro_dir, scenario_condition_dir, condition_dir, video_file)
                        
                        tracking_subdir_path = os.path.join(tracking_data_directory, intro_dir, scenario_condition_dir, condition_dir)
                        if tracking_subdir_path:
                            csv_files = glob.glob(os.path.join(tracking_subdir_path, "*.csv"))

                            j = 0
                            for csv_file in csv_files:
                                csv_path = csv_file
                                prolific_id = os.path.basename(csv_path).split('_')[0]
                                csv_data = pd.read_csv(os.path.join(current_file_dir_path, csv_path), delimiter=";")
                                
                                with process_lock:
                                    output = executor.submit(process_video, j, total_participants_in_condition, video_path, csv_data, csv_path, current_file_dir_path, prolific_id)
                                    futures.append(output)
                                j += 1
