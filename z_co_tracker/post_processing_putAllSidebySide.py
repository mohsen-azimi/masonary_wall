import cv2

# Input video file paths
video_paths = ['assets/output_time_lapse.mp4', 'saved_videos/mask_time_lapse.mp4', 'saved_videos/video_pred_track.mp4']

# Output video file path
output_video_path = 'saved_videos/output_stacked.mp4'

# Initialize video capture objects for each input video
video_captures = [cv2.VideoCapture(path) for path in video_paths]

# Get the dimensions (width and height) of each video
video_dimensions = [(int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))) for cap in video_captures]

# Find the smallest width and height among the videos
min_width = min(dimension[0] for dimension in video_dimensions)
min_height = min(dimension[1] for dimension in video_dimensions)

# Initialize the video writer with the smallest dimensions
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, 30, (min_width * len(video_paths), min_height))

# Loop through frames, resize, and stack horizontally
while True:
    frames = []
    for cap in video_captures:
        ret, frame = cap.read()
        if not ret:
            break
        # Resize the frame to the smallest dimensions
        resized_frame = cv2.resize(frame, (min_width, min_height))
        frames.append(resized_frame)

    # if not any(frames):
    #     break

    # Horizontally stack the frames
    stacked_frame = cv2.hconcat(frames)

    # Write the stacked frame to the output video
    out.write(stacked_frame)

# Release the video objects
for cap in video_captures:
    cap.release()
out.release()

print(f"Videos stacked horizontally and resized to smallest dimensions. Result saved at {output_video_path}")
