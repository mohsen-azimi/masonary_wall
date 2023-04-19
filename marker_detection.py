import os
import exifread
import datetime


import numpy as np
import cv2
import cv2.aruco as aruco
from utils import *


DIR = "./dataset/"

ARUCO_DICT = {
            "DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
        }
# Define the ArUco dictionary
for key, value in ARUCO_DICT.items():
    aruco_dict = cv2.aruco.Dictionary_get(value)
# aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_5X5_50)

    # Define the detector parameters
    # parameters = cv2.aruco.DetectorParameters_create()


    files = [f for f in os.listdir(DIR) if f.endswith('.JPG') and not f.startswith('marker')]

    for file in files:
        print(file)
        #
        # Read in the image file
        img = cv2.imread(os.path.join(DIR, file))



        ## read metadata from image
        # Open image file for reading (binary mode)
        f = open(os.path.join(DIR, file), 'rb')

        # Return Exif tags
        tags = exifread.process_file(f)
        print(tags['EXIF DateTimeOriginal'])

        # get date and time from metadata
        date_time = tags['EXIF DateTimeOriginal']
        date_time = str(date_time)
        date_time = date_time.split(' ')
        date = date_time[0]
        time = date_time[1]
        print(date)
        print(time)



        #
        # # Detect ArUco markers in the image
        # corners, ids, _ = cv2.aruco.detectMarkers(img, aruco_dict)
        # # print("Corners: ", corners)
        # print("IDs: ", ids)
        # print("dict: ", key)
        # # print("Rejected Image Points: ", rejectedImgPoints)
        #
        #
        # # Draw detected markers on the image and display
        # img_with_markers = cv2.aruco.drawDetectedMarkers(img, corners, ids)
        #
        # # Display the image
        # imshow_img(img_with_markers, 1.0)
        # # cv2.imshow("Detected Markers", img_with_markers)
        # # cv2.waitKey(0)

# cv2.destroyAllWindows()
