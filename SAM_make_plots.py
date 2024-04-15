import os

import cv2
import numpy as np
from matplotlib import pyplot as plt

video_frames = []
track_marker = True

# frames_DIR = "masks"
frames_DIR = "C:/Users\Mohsen\Desktop\data_leila\output_masks"
frames = os.listdir(frames_DIR)
frames.sort()
skip = True

import json

class Track_marker():
    def __init__(self):
        self.track = {}

        print("Track_marker initialized")

    def fit_square(self, frame_DIR):

        files = os.listdir(frame_DIR)
        frame_id = frame_DIR.split("/")[-1]
        print(frame_id)
        if frame_id not in self.track:
            self.track[frame_id] = {}  # Create an empty dictionary for the frame_id

        marker_masks = [file for file in files if "marker" in file]

        for mask_file in marker_masks:
            mask = cv2.imread(f"{frame_DIR}/{mask_file}", cv2.IMREAD_COLOR)

            mask_copy = mask.copy()
            binary_mask = cv2.cvtColor(mask_copy, cv2.COLOR_BGR2GRAY)
            binary_mask = cv2.threshold(binary_mask, 0, 255, cv2.THRESH_BINARY)[1]


            contours, hierarchy = cv2.findContours(binary_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            if len(contours) == 0:  # to handle the case where the marker is not visible
                self.track[frame_id][mask_file] = {"x": None, "y": None, "w": None, "h": None}
                continue
            contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(contour)

            self.track[frame_id][mask_file] = {"x": x, "y": y, "w": w, "h": h}

    def save_to_json(self):
        # save the results to a json file
        with open("tracking.json", "w") as outfile:
            json.dump(self.track, outfile)

    def load_from_json_and_plot(self, marker_name='beam'):
        print("load_from_json_and_plot")
        # load the json file and plot the results
        with open("tracking.json", "r") as outfile:
            self.track = json.load(outfile)

        print(self.track.keys())

        history = []
        for frame_id, maskers in self.track.items():
            for maker, marker_info in maskers.items():
                if marker_name in maker:
                    print("frame_id", frame_id, "marker", marker_name)
                    history.append(marker_info['x'])

        plt.plot(history)
        plt.title(f"{marker_name}")
        plt.show()


def track():

    for frame in frames:
        print(frame)
        if skip:
            skip = False
            continue


        tracker.fit_square(f"{frames_DIR}/{frame}")  # to track markers



    tracker.save_to_json()

def load_from_json_and_plot(marker_name):
    tracker.load_from_json_and_plot(marker_name=marker_name)

if __name__ == "__main__":
    tracker = Track_marker()

    # track()

    marker_names = ["big_left", "big_right", "beam", "intop", "inbot", "inleft", "inright"]
    for marker_name in marker_names:
        load_from_json_and_plot(marker_name)

