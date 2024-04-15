import matplotlib
matplotlib.use('TkAgg')

import os
import cv2
import numpy as np
import pandas as pd
import time


from matplotlib import pyplot as plt



frames_DIR = 'C:\\Users\\azimi\OneDrive\Desktop\data\data_XP\TwoStoreyGM3_1'
# frames_DIR = 'E:\Data\data_leila\camera2_video\camera2_out'
mask_file = "mask_marker_story2_refined.png"
imshow_file = "mask_marker_story2_refined.png"
frames = os.listdir(frames_DIR)
frames.sort()
h, w = cv2.imread(f"{frames_DIR}/{frames[0]}/{mask_file}", 0).shape
# Load and process all frames to get the mask edges
min_pixel_indices = np.zeros((len(frames), h), dtype=np.int32)
max_pixel_indices = np.zeros((len(frames), h), dtype=np.int32)

for frame_idx, frame in enumerate(frames[:-1]):
    print(frame)
    mask = cv2.imread(f"{frames_DIR}/{frame}/{mask_file}", 0)
    edges = cv2.Canny(mask, 50, 150, apertureSize=3)
    h, w = mask.shape

    for row in range(h):
        row_pixels = edges[row, :]
        white_pixels = np.where(row_pixels == 255)[0]
        if len(white_pixels) > 0:
            min_pixel_indices[frame_idx, row] = white_pixels[0]
            max_pixel_indices[frame_idx, row] = white_pixels[-1]

# Normalize indices to prevent plotting issues
min_pixel_indices = min_pixel_indices.astype(object)
min_pixel_indices[min_pixel_indices == 0] = None
max_pixel_indices = max_pixel_indices.astype(object)
max_pixel_indices[max_pixel_indices == 0] = None

# Scaling factor (50%)
scale_factor = .5
the_mask = cv2.imread(f"{frames_DIR}/{frames[0]}/{mask_file}", 0)
display_mask = cv2.resize(the_mask, (int(w * scale_factor), int(h * scale_factor)))


# Mouse click callback function
def on_mouse_click(event, x, y, flags, param):
    global ax
    global right_movement
    global left_movement
    global selected_h


    if event == cv2.EVENT_LBUTTONDOWN:
        selected_h = int(y / scale_factor)

        if np.any(max_pixel_indices[:, selected_h] != None):
            right_movement = [-1 * (max_val - max_pixel_indices[0, selected_h]) for max_val in max_pixel_indices[:, selected_h] if max_val is not None]
            left_movement = [-1 * (min_val - min_pixel_indices[0, selected_h]) for min_val in min_pixel_indices[:, selected_h] if min_val is not None]

            # Clear the previous plot
            ax.clear()

            ax.plot(right_movement, label="Right")
            # ax.plot(left_movement, label="Left")
            ax.set_title(f"Mask Motion at h: {selected_h}")
            ax.legend()

            # Redraw the plot
            fig.canvas.draw()
            fig.canvas.flush_events()
            # sleep(0.1)
            # plt.show()
            time.sleep(0.1)
        else:
            print("No valid data to plot.")

    elif event == cv2.EVENT_RBUTTONDOWN:
        # Right click - Save data to CSV

        if 'right_movement' in globals() and 'left_movement' in globals():
            # save the click location as well

            data = {"Right_Movement": right_movement, "Left_Movement": left_movement, "h": selected_h}
            df = pd.DataFrame(data)
            df.to_csv(f"{frames_DIR}\plot_data.csv", index=False)
            print("Data saved to plot_data.csv")



plt.ion()  # Enable interactive mode
fig, ax = plt.subplots()


# Set up the window and callback, and display the image
cv2.namedWindow("Image")
cv2.setMouseCallback("Image", on_mouse_click)
cv2.imshow("Image", display_mask)
cv2.waitKey(0)
cv2.destroyAllWindows()
