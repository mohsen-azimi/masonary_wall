import cv2
import os

def save_frames_from_video(video_path):
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Get video information
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Create a folder to save frames
    folder_name = os.path.splitext(os.path.basename(video_path))[0]
    output_folder = os.path.join(os.path.dirname(video_path), folder_name)
    os.makedirs(output_folder, exist_ok=True)

    # Read and save each frame
    for frame_number in range(frame_count):
        ret, frame = cap.read()
        if not ret:
            break

        # save every 100th frame
        if frame_number % 10 != 0:
            continue
        # reverse the mask
        # frame = cv2.bitwise_not(frame)
        frame_filename = os.path.join(output_folder, f"frame_{frame_number:04d}.png")
        cv2.imwrite(frame_filename, frame)

    # Release the video capture object
    cap.release()

if __name__ == "__main__":
    # Replace 'your_video.mp4' with the path to your MP4 video file
    video_path = 'output_time_lapse.mp4'

    save_frames_from_video(video_path)
