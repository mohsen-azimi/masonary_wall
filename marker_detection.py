import os
import exifread
import datetime
import json
import tqdm
import numpy as np
import cv2
import cv2.aruco as aruco
from utils import *

camera = 'camera2'   # camera2--jpg



DIR = "./dataset/"+camera

ARUCO_DICT = {
    "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
    "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
    "DICT_4X4_250": cv2.aruco.DICT_4X4_250,
    "DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
    "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
    "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
    "DICT_5X5_250": cv2.aruco.DICT_5X5_250,
    "DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
    "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
    "DICT_6X6_100": cv2.aruco.DICT_6X6_100,
    "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
    "DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
    "DICT_7X7_50": cv2.aruco.DICT_7X7_50,
    "DICT_7X7_100": cv2.aruco.DICT_7X7_100,
    "DICT_7X7_250": cv2.aruco.DICT_7X7_250,
    "DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
    "DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
}

ARUCO_DICT = {
    "DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
}
# Define the ArUco dictionary


json_list = []

for key, value in ARUCO_DICT.items():
    aruco_dict = cv2.aruco.Dictionary_get(value)
    # aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_5X5_50)

    # Define the detector parameters
    # parameters = cv2.aruco.DetectorParameters_create()

    files = [f for f in os.listdir(DIR) if f.endswith('.JPG') or f.endswith('.jpg') and not f.startswith('marker')]

    for file in tqdm.tqdm(files):
        # print(file)
        #
        # Read in the image file
        img = cv2.imread(os.path.join(DIR, file))



        if camera == 'canon':
            ## read metadata from image
            # Open image file for reading (binary mode)
            f = open(os.path.join(DIR, file), 'rb')

            # Return Exif tags
            tags = exifread.process_file(f)
            # print(tags['EXIF DateTimeOriginal'])

            # get date and time from metadata
            date_time = tags['EXIF DateTimeOriginal']
            date_time = str(date_time)
            date_time = date_time.split(' ')
            date = date_time[0]
            time = date_time[1]
        elif camera == 'camera2':
            # Parse date and time from string

            date_, time_ = file.split("_")
            date = f"{date_[0:4]}:{date_[4:6]}:{date_[6:8]}"
            time = f"{time_[0:2]}:{time_[2:4]}:{time_[4:6]}"



        #
        # Detect ArUco markers in the image
        corners, ids, _ = cv2.aruco.detectMarkers(img, aruco_dict)
        # print("Corners: ", corners)
        if ids is not None:
            #     print("IDs: ", ids)
            #     print("dict: ", key)
            # print("Rejected Image Points: ", rejectedImgPoints)

            #
            # Draw detected markers on the image and display
            # img_with_markers = cv2.aruco.drawDetectedMarkers(img, corners, ids)

            # Display the image
            # imshow_img(img_with_markers, .15)
            # cv2.imshow("Detected Markers", img_with_markers)
            # cv2.waitKey(0)

            # make a dictionary of the extracted information
            Corners = tuple(c.squeeze().tolist() for c in corners)

            extracted_info = {
                "file_name": file,
                "date": date,
                "time": time,
                "corners": Corners,
                "ids": ids.squeeze().tolist(),
                "dict": key
            }

            # append the dictionary to a list
            json_list.append(extracted_info)

# save the list as a json file
with open(f'./results/marker_detection_{camera}.json', 'w') as f:
    json.dump(json_list, f)

#
# # cv2.destroyAllWindows()
