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
from segment_anything_local import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import winsound
from utils import *
from segmentation_refinement.segmentation_refinement import Refiner

import json

print("PyTorch version:", torch.__version__, "\nTorchvision version:", torchvision.__version__, "\nCUDA is available:",
      torch.cuda.is_available())

# video_file = '../Dataset/PRJ-3446/Videos/S0.0.1.wmv'
video_DIR = 'C:/Users/azimi\OneDrive/Desktop/gDriveBackup\Research/02_SAM_Tracking/z_co-tracker/assets/leila_camera2/camera2.mp4'
OUT_DIR = 'C:\\Users\\azimi\OneDrive\Desktop\data\z_temp_output'
points_DIR = 'C:\\Users\\azimi\OneDrive\Desktop\gDriveBackup\Research\\02_SAM_Tracking\z_co-tracker\saved_videos\camera2.mp4_tracks_and_visibility.npz'
display_points = True
do_refine = True

cap = cv2.VideoCapture(video_DIR)
fps = cap.get(cv2.CAP_PROP_FPS)

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# load json data
# Loading data from the saved .npz file
tracking_data = np.load(points_DIR, allow_pickle=True)
tracks = tracking_data["tracks"]
visibility = tracking_data["visibility"]
#
print("Total frames:", total_frames)
print("FPS:", fps)
print("tracks.shape:", tracks.shape)
print("visibility.shape:", visibility.shape)

# SAM
sam_checkpoint = "weights/sam_vit_h_4b8939.pth"
model_type = "vit_h"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
mask_generator = SamAutomaticMaskGenerator(sam)
mask_predictor = SamPredictor(sam)

refiner = Refiner(device='cuda:0')  # (device='cuda:0')  # device can also be 'cpu'

print("SAM model is ready!\n\n\n")

# Loop through all the frames
frame_id = 0
video_frame = []
current_state = {}
initial_state = {}  # initial state of the object to be tracked
flag = True

masks_frames = []
masks_frames_refined = []

show_masks = False
while True:
    # Read the next frame

    success, frame = cap.read()
    if not success:
        break

    if not os.path.exists(f'{OUT_DIR}/Frame_{frame_id:06d}'):
        os.makedirs(f'{OUT_DIR}/Frame_{frame_id:06d}')

    image_copy = frame.copy()  # for drawing points
    overlay = frame.copy()  # for showing the masks with alpha channel
    overlay_refined = frame.copy()  # for showing the masks with alpha channel
    frame_with_masks = frame.copy()  # for saving the masks
    frame_with_masks_refined = frame.copy()  # for saving the masks

    # TODO: remove this, resized by 20%
    # frame = cv2.resize(frame, (0, 0), fx=.5, fy=0.5)

    h, w, c = frame.shape

    # # save the first frame to a png file under ./masks/
    # if not os.path.exists(f'{OUT_DIR}/Frame_{frame_id:06d}'):
    #     os.makedirs(f'{OUT_DIR}/Frame_{frame_id:06d}')

    # cv2.imwrite(f'{OUT_DIR}/Frame_{frame_id:06d}/frame.png', frame)

    mask_predictor.set_image(frame)

    #
    points_visibility = visibility[frame_id] # [1,1, 1, 1, 1, 1]
    points_coords = tracks[frame_id]

    # only keep the visible points
    points_coords = points_coords[points_visibility == 1]
    print(f"points_coords: {points_coords.shape}, points_visibility: {points_visibility.shape}")

    display_points = 0
    if display_points:
        for point in points_coords:
            x, y = int(point[0]), int(point[1])
            cv2.circle(image_copy, (x, y), 50, (0, 0, 255), -1)
            cv2.putText(image_copy, f"({x}, {y})", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 4)

        cv2.imshow("Points", cv2.resize(image_copy, (0, 0), fx=0.25, fy=0.25))
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # Now segment the object using the points

    """
    point_coords (np.ndarray or None): A Nx2 array of point prompts to the
    model. Each point is in (X,Y) in pixels.
    point_labels (np.ndarray or None): A length N array of labels for the
    point prompts. 1 indicates a foreground point and 0 indicates a
    background point.
    """

    masks, _, _ = mask_predictor.predict(point_coords=points_coords,
                                         point_labels= np.ones((points_coords.shape[0],), dtype=np.int32),
                                         multimask_output=False,
                                         mask_input=None)

    print(f"|------ Masks generated for frame {frame_id}/{total_frames} ------|")
    # show the mask
    if show_masks:
        for mask in masks:
            cv2.imshow(f"mask/{len(masks)}", cv2.resize(mask.astype(np.uint8) * 255, (0, 0), fx=0.2, fy=0.2))
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    mask = masks[0] * 255
    mask = mask.astype(np.uint8)

    cv2.imwrite(f'{OUT_DIR}/Frame_{frame_id:06d}/mask.png', mask)

    if do_refine:
        mask_refined = refiner.refine(frame, mask, fast=False, L=900)
        mask_refined = (mask_refined > 127).astype(np.uint8) * 255
        cv2.imwrite(f'{OUT_DIR}/Frame_{frame_id:06d}/mask_refined.png', mask_refined)

        masks_frames_refined.append(mask_refined)
    masks_frames.append(mask)
    frame_id += 1
    # if frame_id > 10:
    #     break
    # end of the loop for all frames


# save the masks as mp4
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
h, w= mask.shape
print(f"masks[0].shape: {masks.shape}")
out = cv2.VideoWriter(f'{OUT_DIR}/all_masks.mp4', fourcc, 10, (w, h))
for m in masks_frames:
    m_bgr = cv2.cvtColor(m, cv2.COLOR_GRAY2BGR)
    out.write(m_bgr)
out.release()


out_refined = cv2.VideoWriter(f'{OUT_DIR}/all_masks_refiend.mp4', fourcc, 10, (w, h))
for m in masks_frames_refined:
    m_bgr = cv2.cvtColor(m, cv2.COLOR_GRAY2BGR)
    out_refined.write(m_bgr)
out_refined.release()

print("Done!")
