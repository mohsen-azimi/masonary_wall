# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os

import cv2
import torch
import argparse
import numpy as np

from PIL import Image
from matplotlib import pyplot as plt

from cotracker.utils.visualizer import Visualizer, read_video_from_path
from cotracker.predictor import CoTrackerPredictor

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

DEFAULT_DEVICE = ('cuda' if torch.cuda.is_available() else
                  'mps' if torch.backends.mps.is_available() else
                  'cpu')
print(f"Using device: {DEFAULT_DEVICE}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--video_path",
        default="./assets/leila_camera2/camera2.mp4",
        help="path to a video",
    )
    parser.add_argument(
        "--mask_path",
        default="./assets/leila_camera2/mask_frame_merged.png",
        help="path to a segmentation mask",
    )
    parser.add_argument(
        "--checkpoint",
        default="./checkpoints/cotracker_stride_4_wind_8.pth",
        help="cotracker model",
    )
    parser.add_argument("--grid_size", type=int, default=500, help="Regular grid size")
    parser.add_argument(
        "--grid_query_frame",
        type=int,
        default=0,
        help="Compute dense and grid tracks starting from this frame ",
    )

    parser.add_argument(
        "--backward_tracking",
        action="store_true",
        help="Compute tracks in both directions, not only forward",
    )

    args = parser.parse_args()

    # load the input video frame by frame
    video = read_video_from_path(args.video_path)
    print(f"video shape: {video.shape}")
    video = torch.from_numpy(video).permute(0, 3, 1, 2)[None].float()

    segm_mask = np.array(Image.open(os.path.join(args.mask_path)))
    # display the mask
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

    # keep only at h=h_min: h_max, the rest is zero
    h_min, h_max = 100, 105
    segm_mask[:h_min] = 0
    segm_mask[h_max:] = 0
    plt.imshow(segm_mask)
    plt.show()



    segm_mask = torch.from_numpy(segm_mask)[None, None]
    # segm_mask = None

    model = CoTrackerPredictor(checkpoint=args.checkpoint)
    model = model.to(DEFAULT_DEVICE)
    video = video.to(DEFAULT_DEVICE)

    pred_tracks, pred_visibility = model(
        video,
        grid_size=args.grid_size,
        grid_query_frame=args.grid_query_frame,
        backward_tracking=args.backward_tracking,
        segm_mask=segm_mask
    )
    print("computed")

    # save a video with predicted tracks
    seq_name = args.video_path.split("/")[-1]
    vis = Visualizer(save_dir="./saved_videos", pad_value=0, linewidth=1, fps=1, tracks_leave_trace=0, mode="optical_flow")
    # vis = Visualizer(save_dir="./saved_videos", pad_value=0, linewidth=2, fps=60, tracks_leave_trace=0, mode="cool")
    # video.fill_(255)
    vis.visualize(video, pred_tracks, pred_visibility, query_frame=args.grid_query_frame)

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
        f"./saved_videos/{seq_name}_tracks_and_visibility.npz",
        tracks=tracks,
        visibility=visibility,
    )


