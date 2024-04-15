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
configs = yaml.safe_load(open("configs.yaml"))
print(configs["coTracker"])

video_path = configs["coTracker"]["video_path"]
mask_path = configs["coTracker"]["mask_path"]
checkpoint_path = configs["coTracker"]["checkpoint_path"]
output_path = configs["coTracker"]["output_path"]
os.makedirs(output_path, exist_ok=True)


grid_size = configs["coTracker"]["grid_size"]
grid_query_frame = configs["coTracker"]["grid_query_frame"]
backward_tracking = configs["coTracker"]["backward_tracking"]

crop_y0 = configs["coTracker"]["crop_y0"]
crop_y1 = configs["coTracker"]["crop_y1"]
do_crop = True if crop_y1 != -1 else False



##########################################

video = read_video_from_path(video_path)
if do_crop:
    video = video[:, crop_y0:crop_y1, :, :]
print(f"video shape: {video.shape}")
video = torch.from_numpy(video).permute(0, 3, 1, 2)[None].float()

segm_mask = np.array(Image.open(os.path.join(mask_path)))
if do_crop:
    segm_mask = segm_mask[crop_y0:crop_y1, :]
plt.imshow(segm_mask)
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

# # keep only at h=h_min: h_max, the rest is zero
# h_min, h_max = 100, 105
# segm_mask[:h_min] = 0
# segm_mask[h_max:] = 0
# plt.imshow(segm_mask)
# plt.show()



segm_mask = torch.from_numpy(segm_mask)[None, None]
# segm_mask = None

interp_shape = (video.shape[3], video.shape[4]) if do_crop else (384, 512)  # torch.Size([1, n, C, h, w])
print(interp_shape)

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
np.savez(
    f"{output_path}/{seq_name}_tracks_and_visibility.npz",
    tracks=tracks,
    visibility=visibility,
)



