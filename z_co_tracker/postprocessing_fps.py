import cv2
import numpy as np

# Open the video file
video_file = r'C:\Users\Mohsen\Desktop\tracking_outputs\S0.0.2.mp4'
cap = cv2.VideoCapture(video_file)

# Check if the video file was successfully opened
if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()

# Get the frames per second (fps) of the original video
fps_in = int(cap.get(cv2.CAP_PROP_FPS))
print(f"Original video fps = {fps_in}")
print(f"Original frame size = {int(cap.get(3))} x {int(cap.get(4))}")

h_old, w_old = int(cap.get(4)), int(cap.get(3))
fps_out = 30  # Define the desired output fps

# Define the output file name
output_file = './saved_videos/mask_time_lapse.mp4'

# Define the codec and create a VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # You can change the codec as needed
out = cv2.VideoWriter(output_file, fourcc, fps_out, (int(w_old / 2), int(h_old / 2)))

# Define the frame interval (e.g., 10 for every 10th frame)
frame_interval = 8

frame_count = 0

while True:
    # Set the frame position to the desired frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)

    ret, frame = cap.read()

    if not ret or frame_count>8000:
        break  # Break the loop when we reach the end of the video

    # resize the frame
    h_new, w_new = int(frame.shape[0] / 2), int(frame.shape[1] / 2)
    frame = cv2.resize(frame, (w_new, h_new))




    print(f"Frame {frame_count} / {int(cap.get(cv2.CAP_PROP_FRAME_COUNT))}")
    out.write(frame)  # Write the frame to the output video

    frame_count += frame_interval

# Release video capture and writer objects
cap.release()
out.release()

# Close all OpenCV windows
cv2.destroyAllWindows()

print(f"Time-lapse video saved as {output_file}")

