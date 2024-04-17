# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import os
import cv2
import torch
import argparse
import numpy as np
import yaml
from PIL import Image
from matplotlib import pyplot as plt
from cotracker.utils.visualizer import Visualizer, read_video_from_path
from cotracker.predictor import CoTrackerPredictor
from utils import print_frame_count
# ........................................
from git_update import git_add_commit_push
git_add_commit_push()
##########################################

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

DEFAULT_DEVICE = ('cuda' if torch.cuda.is_available() else
                  'mps' if torch.backends.mps.is_available() else
                  'cpu')
print(f"Using device: {DEFAULT_DEVICE}")


# read the config yaml file
configs = yaml.safe_load(open("input_output/inputs/wood/coTracker_configs.yaml"))


video_path = configs["coTracker"]["video_path"]
mask_path = configs["coTracker"]["mask_path"]
checkpoint_path = configs["coTracker"]["checkpoint_path"]
output_path = configs["coTracker"]["output_path"]
os.makedirs(output_path, exist_ok=True)


grid_size = configs["coTracker"]["grid_size"]
grid_query_frame = configs["coTracker"]["grid_query_frame"]
backward_tracking = configs["coTracker"]["backward_tracking"]

crop_frame_y0 = configs["coTracker"]["crop_frame_y0"]
crop_frame_y1 = configs["coTracker"]["crop_frame_y1"]
crop_frame_x0 = configs["coTracker"]["crop_frame_x0"]
crop_frame_x1 = configs["coTracker"]["crop_frame_x1"]

mask_strip_y0 = configs["coTracker"]["mask_strip_y0"]
mask_strip_y1 = configs["coTracker"]["mask_strip_y1"]
cut_strip = True if mask_strip_y1 != -1 else False

start_frame = configs["coTracker"]["start_frame"]
end_frame = configs["coTracker"]["end_frame"]

##########################################
print(print_frame_count(video_path))

video = read_video_from_path(video_path, start_frame=start_frame, end_frame=end_frame,
                             crop_roi=(crop_frame_y0,crop_frame_y1, crop_frame_x0,crop_frame_x1))

print(f"video shape (cropped): {video.shape}")
video = torch.from_numpy(video).permute(0, 3, 1, 2)[None].float()

segm_mask = np.array(Image.open(os.path.join(mask_path)))

if cut_strip:
    print(f"cutting the strip: {mask_strip_y0} - {mask_strip_y1}")
    segm_mask[:mask_strip_y0] = 0
    segm_mask[mask_strip_y1:] = 0
    plt.imshow(segm_mask)
    plt.title("cut strip")
    plt.show()

segm_mask = segm_mask[crop_frame_y0:crop_frame_y1, crop_frame_x0:crop_frame_x1]
plt.imshow(segm_mask)
plt.title("segm_mask")
plt.show()



# fill the largest contour in the mask
contours, _ = cv2.findContours(segm_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
segm_mask = np.zeros_like(segm_mask)
cv2.drawContours(segm_mask, contours, -1, 255, -1)
plt.imshow(segm_mask)
plt.show()

# # erode the mask to keep the points inside the boundary of the mask
# kernel = np.ones((15, 15), np.uint8)
# segm_mask = cv2.erode(segm_mask, kernel, iterations=1)
# # display the mask
# plt.imshow(segm_mask)
# plt.show()




segm_mask = torch.from_numpy(segm_mask)[None, None]
# segm_mask = None

interp_shape = (384, 512)
if crop_frame_x1 != -1 and crop_frame_y1 != -1:
    interp_shape = (video.shape[3], video.shape[4]) # torch.Size([1, n, C, h, w])

print(f"interp_shape: {interp_shape}")

model = CoTrackerPredictor(checkpoint=checkpoint_path, interp_shape=interp_shape)
model = model.to(DEFAULT_DEVICE)
video = video.to(DEFAULT_DEVICE)

pred_tracks, pred_visibility = model(
    video,
    grid_size=grid_size,
    grid_query_frame=grid_query_frame,
    backward_tracking=backward_tracking,
    segm_mask=segm_mask
)
print("computed")
print(f"pred_tracks.shape: {pred_tracks.shape}"
      f"pred_visibility.shape: {pred_visibility.shape}")

# save a video with predicted tracks
seq_name = video_path.split("/")[-1]
vis = Visualizer(save_dir=output_path, pad_value=0, linewidth=1, fps=1, tracks_leave_trace=0, mode="optical_flow")
# vis = Visualizer(save_dir="./saved_videos", pad_value=0, linewidth=2, fps=60, tracks_leave_trace=0, mode="cool")
# video.fill_(255)
vis.visualize(video, pred_tracks, pred_visibility, query_frame=grid_query_frame)

print(f"pred_tracks.shape: {pred_tracks.shape}")
print(f"pred_visibility.shape: {pred_visibility.shape}")

# pred_tracks.shape: torch.Size([1, 492, 2, 2])
# print(pred_tracks[0,0,:,:])
tracks = pred_tracks[0].long().detach().cpu().numpy() # S, N, 2
visibility = pred_visibility[0].long().detach().cpu().numpy() # S, N
print(tracks.shape)
print(visibility.shape)
# print(tracks)

# save the tracks and visibility to a file (format=npz)
points_DIR = f"{output_path}/{seq_name}_tracks_and_visibility.npz"
np.savez(
    f"{points_DIR}",
    tracks=tracks,
    visibility=visibility,
)



# ############## Now plot the tracks and visibility


# load json data
# Loading data from the saved .npz file
tracking_data = np.load(points_DIR, allow_pickle=True)
tracks = tracking_data["tracks"]
# multiply by an scale factor
scale_factor = configs["coTracker"]["calibration_scale_factor"]
tracks = tracks*scale_factor
visibility = tracking_data["visibility"]
#

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

