"""Segment_anything.ipynb    https://colab.research.google.com/drive/1VeU6OWmrWylZ3568BsUXGlO8N9Fyv0CQ
 pip install 'git+https://github.com/facebookresearch/segment-anything.git'



segmentation - [np.ndarray] - the mask with (W, H) shape, and bool type, where W and H are the width and height of the original image, respectively
area - [int] - the area of the mask in pixels
bbox - [List[int]] - the boundary box detection in xywh format
predicted_iou - [float] - the model's own prediction for the quality of the mask
point_coords - [List[List[float]]] - the sampled input point that generated this mask
stability_score - [float] - an additional measure of mask quality
crop_box - List[int] - the crop of the image used to generate this mask in xywh format

"""
import torch  # conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import cv2
import sys
import os

import json


points_DIR = 'C:\\Users\Mohsen\Desktop\gDriveBackup\Research\\02_SAM_Tracking\z_co-tracker\saved_videos\camera2.mp4_tracks_and_visibility.npz'



# load json data
# Loading data from the saved .npz file
tracking_data = np.load(points_DIR, allow_pickle=True)
tracks = tracking_data["tracks"]
# multiply by an scale factor
scale_factor = 4.6603  # pixel to mm
tracks = tracks*scale_factor
visibility = tracking_data["visibility"]
#
print("tracks.shape:", tracks.shape)
print("visibility.shape:", visibility.shape)

# plot the x-displacement of the tracked points
# size of figure is 10x6
fig, ax = plt.subplots(figsize=(10, 6), dpi=600)
ax.set_title("X-displacement of the tracked points")
ax.set_xlabel("Frame")
ax.set_ylabel("X-displacement")
ax.grid(True)

# add linear correlation to the data
for track_idx in range(tracks.shape[1]):
    end_offset = tracks[:, track_idx, 0][-1] - tracks[:, track_idx, 0][0]
    correction_array = np.linspace(0, end_offset, num=tracks.shape[0])
    tracks[:, track_idx, 0] = tracks[:, track_idx, 0] - correction_array


for track_idx in range(tracks.shape[1]):

    ax.plot(tracks[:, track_idx, 0]-tracks[0, track_idx, 0], label=f"Point {track_idx}", linewidth=.5)

# plot the mean of all the x-displacements
mean_x_displacement = np.mean(tracks[:, :, 0], axis=1)
ax.plot(mean_x_displacement-mean_x_displacement[0], label="Mean", linewidth=1.5, color="black")

# plot the median of all the x-displacements
median_x_displacement = np.median(tracks[:, :, 0], axis=1)
ax.plot(median_x_displacement-median_x_displacement[0], label="Median", linewidth=1, color="red")


plt.show()
# ####################################################################################################################
# plot mean and median of the x-displacement in a new figure
fig, ax = plt.subplots(figsize=(10, 6), dpi=600)
ax.set_title("Mean and Median of the X-displacement")
ax.set_xlabel("Frame")
ax.set_ylabel("X-displacement")
ax.grid(True)

ax.plot(mean_x_displacement-mean_x_displacement[0], label="Mean", linewidth=1.5, color="black")
ax.plot(median_x_displacement-median_x_displacement[0], label="Median", linewidth=1, color="red")

plt.legend()
plt.show()


print("Done!")