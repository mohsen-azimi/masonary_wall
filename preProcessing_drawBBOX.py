import cv2
import numpy as np

# Callback function for the trackbars
def nothing(x):
    pass

# Initialize the box parameters
box_params = {"x": 100, "y": 100, "width": 50, "height": 50}

# Initialize the window and trackbars
cv2.namedWindow('control')
cv2.createTrackbar("X", "control", box_params["x"], 1500, nothing)
cv2.createTrackbar("Y", "control", box_params["y"], 2500, nothing)
cv2.createTrackbar("Width", "control", box_params["width"], 2000, nothing)
cv2.createTrackbar("Height", "control", box_params["height"], 2200, nothing)

# Load the image frame of a video
# video_path = 'C:\\Users\\azimi\OneDrive\Desktop\data\\vancouver_house/vancouver_house.mp4'
video_path = './inputs\camera2/camera2.mp4'
# # video_path = 'E:\Data\SAM_tracking\Javadinasab\PRJ-3446\Videos\S0.0.1.wmv'
# video_path = 'C:\\Users\\azimi\OneDrive\Desktop\Zhong_dataverse_files\Videos\T242\\T242_Overview.mp4'
cap = cv2.VideoCapture(video_path)
ret, image = cap.read()
# image_path = "E:/Data/data_leila/lorex_2/N882A6_ch2_main_20230412103151_20230412105916_out/Frame_000000/frame.png"
# image_path = "E:\Data/data_leila/Cannon-DSLR-EOS Rebel T7i/DCIM/100CANON/IMG_0061.JPG"
# image = cv2.imread(image_path)
h, w, _ = image.shape
print(h, w)

# Resize the image to fit the window if necessary

while True:
    # Get the current positions of the trackbars
    x = cv2.getTrackbarPos("X", "control")
    y = cv2.getTrackbarPos("Y", "control")
    width = cv2.getTrackbarPos("Width", "control")
    height = cv2.getTrackbarPos("Height", "control")

    # Make a copy of the resized image to draw the box
    img_with_box = image.copy()

    # Draw the box on the image
    cv2.rectangle(img_with_box, (x, y), (x + width, y + height), (0, 0, 255), 2)

    # Display the image with the box
    tmp_resized = cv2.resize(img_with_box, (int(w * 1 / 1), int(h * 1 / 1)))
    cv2.imshow("Image with Box", tmp_resized)

    # Break the loop when the 'Esc' key is pressed
    if cv2.waitKey(1) & 0xFF == 27:
        break

cv2.destroyAllWindows()
