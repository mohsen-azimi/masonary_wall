# this code merges three binary masks

import cv2

masks_paths=["./assets/leila_camera2/mask_frame_infill.png",
             "./assets/leila_camera2/mask_frame_left_col.png",
             "./assets/leila_camera2/mask_frame_right_col.png"]

masks = [cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) for mask_path in masks_paths]

# merge the masks
merged_mask = cv2.bitwise_or(masks[0], masks[1])
merged_mask = cv2.bitwise_or(merged_mask, masks[2])

# type cast the mask to uint8
merged_mask = merged_mask.astype('uint8')*255
# save the merged mask
cv2.imwrite("./assets/leila_camera2/mask_frame_merged.png", merged_mask)
