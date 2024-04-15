import cv2

# Input video file path
input_video_path = 'saved_videos/video_pred_track.mp4'

# Output video file path
output_video_path = 'saved_videos/video_pred_track_cropped.mp4'

# Initialize the video capture object
cap = cv2.VideoCapture(input_video_path)

# Read the first frame to get its size and detect white margins
ret, frame = cap.read()
if not ret:
    raise ValueError("Failed to read the first frame.")

# Define the color for white (assuming a white background)
white_color = (255, 255, 255)

# Find the coordinates of the non-white region in the first frame
non_white_coords = cv2.findNonZero(cv2.inRange(frame, white_color, white_color))

# Calculate the bounding box of the non-white region
x, y, w, h = cv2.boundingRect(non_white_coords)

# Initialize the video writer with the desired resolution
output_width, output_height = 1080, 1920
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, 30, (output_width, output_height))

# Loop through the frames, crop, and resize
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Crop the frame to remove the white margins
    cropped_frame = frame[y:y+h, x:x+w]

    # Resize the cropped frame to the desired resolution
    resized_frame = cv2.resize(cropped_frame, (output_width, output_height))

    # Write the resized frame to the output video
    out.write(resized_frame)

# Release the video objects
cap.release()
out.release()

print(f"Video saved with cropped and resized dimensions at {output_video_path}")
