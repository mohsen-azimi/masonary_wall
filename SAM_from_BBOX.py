
import torch  # conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import cv2
import sys
import os
from segment_anything_local import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
# from segment_anything_hq_local.segment_anything_hq import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

import winsound
from utils import *
from segmentation_refinement.segmentation_refinement import Refiner

import json
from tqdm import tqdm

print("PyTorch version:", torch.__version__, "\nTorchvision version:", torchvision.__version__, "\nCUDA is available:",
      torch.cuda.is_available())


video_DIR = './input_output/inputs/wood/'
video_file_name = 'N882A6_ch2_main_20221012110243_20221012110912.mp4' # light_2021_10_07_14_53_IMG_9159_RSN52_1, light_2021_10_07_14_35_IMG_9157_RSN81_1
frame_0_json = 'N882A6_ch2_main_20221012110243_20221012110912.json'
file_type = '.mp4'


display_bbox = False
do_refine = True
do_crop = False
imwrite_frame = True
skip_frames = False

# keep only those with .wmv extension or mp4
#

video_file = os.path.join(video_DIR, video_file_name)
print("video_file:", video_file)
OUT_DIR = f"{video_DIR}/{video_file_name.replace(file_type, '')}"
print("OUT_DIR:", OUT_DIR)

cap = cv2.VideoCapture(video_file)
fps = cap.get(cv2.CAP_PROP_FPS)
if skip_frames:
    skip_every_n_frames = 8
else:
    skip_every_n_frames = 1

print("FPS:", fps)

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Check if the video file was successfully opened
if not cap.isOpened():
    print('Error opening video file:', video_file)
    exit()

# load json data
with open(os.path.join(video_DIR, frame_0_json)) as f:
    bbox_data = json.load(f)
    # Dynamically get the last 'region' and its last 'item'
    if bbox_data:
        last_region = list(bbox_data.keys())[-1]  # Get the last region key
        last_item = list(bbox_data[last_region].keys())[-1]  # Get the last item key of the last region

# SAM
tmporary_frame_id = None
sam_checkpoint = "checkpoints/sam_vit_h_4b8939.pth"
model_type = "vit_h"
# sam_checkpoint = "weights/sam_hq_vit_h.pth"
# model_type = "vit_h"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
mask_generator = SamAutomaticMaskGenerator(sam)
mask_predictor = SamPredictor(sam)

refiner = Refiner(device='cuda:0')  # (device='cuda:0')  # device can also be 'cpu'

print("SAM model is ready!\n\n\n")
time.sleep(5)

# Loop through all the frames
frame_id = 0
video_frame = []
current_state = {}
initial_state = {}  # initial state of the object to be tracked
flag = True

with tqdm(total=total_frames, desc=f"{video_file.replace('.wmv', '')}") as pbar:
    while True:
        # Read the next frame

        success, frame = cap.read()

        # Break the loop if no more frames are available
        if not success or frame is None:
            print('No success!')
            break

        image_copy = frame.copy()  # for drawing bounding boxes
        overlay = frame.copy()  # for showing the masks with alpha channel
        overlay_refined = frame.copy()  # for showing the masks with alpha channel
        frame_with_masks = frame.copy()  # for saving the masks
        frame_with_masks_refined = frame.copy()  # for saving the masks

        # TODO: remove this, resized by 20%
        # frame = cv2.resize(frame, (0, 0), fx=.5, fy=0.5)

        h, w, c = frame.shape

        # skip every n frames

        # skip every n frames
        if skip_frames:
            if frame_id % skip_every_n_frames != 0:
                # print("Skipping frame:", frame_id)
                frame_id += 1
                pbar.update(1)
                continue
            # if frame_id <909:
            #     # print("Skipping frame:", frame_id)
            #     frame_id += 1
            #     pbar.update(1)
            #     continue

        if (total_frames - frame_id) < 1:
            # print("Skipping frame:", frame_id)
            frame_id += 1
            continue
        # else:
        #     print(f"{video_files[video_index].replace('.wmv', '')}: Frame {frame_id} / {total_frames}")

        # save the first frame
        # save the first frame to a png file under ./masks/
        if not os.path.exists(f'{OUT_DIR}/Frame_{frame_id:06d}'):
            os.makedirs(f'{OUT_DIR}/Frame_{frame_id:06d}')

        if imwrite_frame:

            cv2.imwrite(f'{OUT_DIR}/Frame_{frame_id:06d}/frame.png', frame)

        for region, value in bbox_data.items():
            for item, bbox in value.items():
                # print("|------", item, bbox)
                box = np.array([
                    bbox['x'],
                    bbox['y'],
                    bbox['x'] + bbox['width'],
                    bbox['y'] + bbox['height']
                ])

                if display_bbox:
                    # show the image with the bounding box
                    image_copy = cv2.rectangle(image_copy, (box[0], box[1]), (box[2], box[3]), bbox['color'], 2)
                    # print(f"Region: {region}, Item: {item}, Bbox: {box}")
                    if region == last_region and item == last_item:
                        image_copy = cv2.resize(image_copy, (0, 0), fx=0.2, fy=0.2)

                        # add frame id to the image
                        cv2.putText(image_copy, f"Frame {frame_id} / {total_frames}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                        # update the imshow
                        cv2.imshow("Frame", image_copy)
                        cv2.waitKey(1)

                mask_predictor.set_image(frame)

                #
                if tmporary_frame_id is None:
                    sam_mask_input = None
                else:
                    sam_mask_input = None
                    # # print(f"Loading mask input from Frame_{tmporary_frame_id:06d}/mask_{region}_{item}.png")
                    # sam_mask_input = cv2.imread(f'{OUT_DIR}/Frame_{tmporary_frame_id:06d}/mask_{region}_{item}.png', 0)
                    #
                    # # cv2.imshow(f"mask_input{tmporary_frame_id}", mask_input)
                    # # cv2.waitKey(0)
                    # # cv2.destroyAllWindows()
                    # #
                    #
                    # sam_mask_input = generate_sam_mask(sam_mask_input)

                # if region == 'frame' and item == 'beam':
                #     multi_mask_output = True
                # else:
                #     multi_mask_output = False

                multi_mask_output = False  # TODO: remove this

                masks, _, _ = mask_predictor.predict(box=box, multimask_output=multi_mask_output, mask_input=sam_mask_input)

                # calculate the area of the mask for frame 0
                if frame_id == 0:
                    roi_area = masks[0].sum()
                    # print(f"roi_area: {roi_area}")

                # save the masks
                mask = masks[0] * 255
                mask = mask.astype(np.uint8)

                cv2.imwrite(f'{OUT_DIR}/Frame_{frame_id:06d}/mask_{region}_{item}.png', mask)

                if do_refine:
                    mask_refined = refiner.refine(frame, mask, fast=False, L=1300)
                    mask_refined = (mask_refined > 127).astype(np.uint8) * 255
                    cv2.imwrite(f'{OUT_DIR}/Frame_{frame_id:06d}/mask_{region}_{item}_refined.png', mask_refined)

                # show the mask with alpha channel on the frame using cv2
                overlay[mask == 255] = bbox['color']
                if do_refine:
                    overlay_refined[mask_refined == 255] = bbox['color']

                # Updated the bbox for the next frame
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                # Initialize variables for the overall bounding box
                x_min, y_min, x_max, y_max = float('inf'), float('inf'), 0, 0

                if len(contours) > 0:
                    # Find the bounding box of the largest contour
                    for contour in contours:
                        # Find the bounding box of each contour
                        x, y, w, h = cv2.boundingRect(contour)

                        # Update the overall bounding box coordinates
                        x_min = min(x_min, x)
                        y_min = min(y_min, y)
                        x_max = max(x_max, x + w)
                        y_max = max(y_max, y + h)

                        w_max = x_max - x_min
                        h_max = y_max - y_min

                    # update only if the changes are less than 10% of the original size
                    max_change = 50 # pixels
                    bbox_data[region][item]['x'] = x_min if abs(x_min - bbox['x']) < max_change else bbox['x']
                    bbox_data[region][item]['y'] = y_min  if abs(y_min - bbox['y']) < max_change else bbox['y']
                    bbox_data[region][item]['width'] = w_max  if abs(w_max - bbox['width']) < max_change else bbox['width']
                    bbox_data[region][item]['height'] = h_max  if abs(h_max - bbox['height']) < max_change else bbox['height']

        alpha = 0.5
        # frame_with_masks = cv2.addWeighted(overlay, alpha, frame_with_masks, 1 - alpha, 0, frame_with_masks)
        # # save the frame with masks
        # cv2.imwrite(f'{OUT_DIR}/Frame_{frame_id:06d}/frame_with_masks.png', frame_with_masks)

        # if do_refine:
        #     frame_with_masks_refined = cv2.addWeighted(overlay_refined, alpha, frame_with_masks_refined, 1 - alpha, 0, frame_with_masks_refined)
        #     # save the frame with masks
        #     cv2.imwrite(f'{OUT_DIR}/Frame_{frame_id:06d}/frame_with_masks_refined.png', frame_with_masks_refined)
        tmporary_frame_id = frame_id

        frame_id += 1
        flag = False

        # set the pbar_text as frame_id
        pbar.set_description(f"Frame {frame_id} / {total_frames}")
        pbar.update(skip_every_n_frames)

        #



    cap.release()
    cv2.destroyAllWindows()
