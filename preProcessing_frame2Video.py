import cv2
import os

# image_folder = 'E:\\Data\\data_leila\\Cannon-DSLR-EOS Rebel T7i\\DCIM\\100CANON'  # Replace with the path to your folder containing images
# video_name = 'E:\\Data\\data_leila\\CANON_Video\\out.mp4'

image_folder = 'E:\\Data\\data_leila\\camera2'  # Replace with the path to your folder containing images
video_name = 'E:\\Data\\data_leila\\camera_video\\camera2.mp4'

images = [img for img in os.listdir(image_folder) if img.endswith(".jpg")]
images.sort()  # Sort the images if necessary

# Determine the width and height from the first image
frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = frame.shape

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or 'XVID'
video = cv2.VideoWriter(video_name, fourcc, 1, (width, height))

for image in images:
    video.write(cv2.imread(os.path.join(image_folder, image)))
    print(f" processing {image} ...")

cv2.destroyAllWindows()
video.release()

print("Done!")
