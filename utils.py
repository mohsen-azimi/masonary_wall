# utils for image processing
import cv2   # pip install opencv-python
import numpy as np
from scipy.ndimage import label 
import matplotlib.pyplot as plt # pip install matplotlib
import open3d as o3d



def imshow_mask(img, percent):
    height, width = img.shape[:2]
    scale_percent = int(percent*100)  # desired percentage of original size

    width = int(width * scale_percent / 100)
    height = int(height * scale_percent / 100)

    dim = (width, height)

    resized_img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    cv2.imshow("Resized Mask", resized_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

def imshow_img(img, percent):
    height, width = img.shape[:2]
    scale_percent = int(percent*100)  # desired percentage of original size

    width = int(width * scale_percent / 100)
    height = int(height * scale_percent / 100)

    dim = (width, height)

    resized_img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    cv2.imshow("Resized Image", resized_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



def largest_submask_fill(mask):


    # find the largest connected component in the binary mask
    labeled_mask, num_labels = label(mask)
    largest_label = 0
    largest_size = 0
    for i in range(1, num_labels + 1):
        size = np.sum(labeled_mask == i)
        if size > largest_size:
            largest_size = size
            largest_label = i

    # create a binary mask for the largest connected component
    largest_component = (labeled_mask == largest_label)

    # fill the holes inside the largest connected component
    result = np.copy(largest_component)
    result[np.logical_not(mask)] = 0

    return result



def apply_mask(mask_2, mask_3):

    result = np.copy(mask_2)
    result[mask_3 == 1] = 0

    return result

def plot3d(stacked_masks):
    non_zero_indices = stacked_masks.nonzero()
    z = stacked_masks[non_zero_indices]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(non_zero_indices[0], non_zero_indices[1], non_zero_indices[2], c='blue', marker='.', s=1, alpha=0.05)

    ax.set_xlabel('Height')
    ax.set_ylabel('Width')
    ax.set_zlabel('Mask Number')
    ax.set_aspect('equal')

    plt.show()



def save_to_ply(stacked_masks):
    non_zero_indices = stacked_masks.nonzero()
    x = non_zero_indices[0]
    y = non_zero_indices[1]
    z = non_zero_indices[2]

    points = np.column_stack((x, y, z))

    with open("3d.ply", "w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {points.shape[0]}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("end_header\n")

        for point in points:
            f.write(f"{point[0]} {point[1]} {point[2]}\n")


def show_ply(file_path):
    pcd = o3d.io.read_point_cloud(file_path)
    o3d.visualization.draw_geometries([pcd])



def get_submasks(mask):
    submasks = []
    mask_copy = mask.copy()

    # find all connected components in the mask
    num_labels, labels = cv2.connectedComponents(mask_copy)

    # loop through each connected component and create a submask
    for label in range(1, num_labels):
        # create a new mask with only the current connected component
        submask = np.zeros_like(mask_copy)
        submask[labels == label] = 1

        submasks.append(submask)

    return submasks


def colorize_mask(mask):
    # Define color map for each label
    colors = {0: [0, 0, 0],  # black for label 0
              1: [0, 0, 255],  # red for label 1: GEP
              2: [0, 255, 0],  # green for label 2: LOF
              3: [255, 255, 255]}  # yellow for label 3: KH
    # Create RGB array from mask and color map
    rgb = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    for label in colors:
        rgb[mask == label] = colors[label]

    return rgb




def find_farthest_points(cnt, mask):
    # compute pairwise distances between points on the contour
    dists = cv2.distanceTransform(cv2.bitwise_not(cv2.drawContours(
        np.zeros_like(mask), [cnt], 0, 255, cv2.FILLED)), cv2.DIST_L2, 5)

    # find the farthest points on the contour
    max_idx = np.unravel_index(np.argmax(dists), dists.shape)
    farthest_point1 = tuple(max_idx[::-1])
    dists[max_idx] = 0
    max_idx = np.unravel_index(np.argmax(dists), dists.shape)
    farthest_point2 = tuple(max_idx[::-1])

    return farthest_point1, farthest_point2


def fit_ellipse(mask, farthest_point1, farthest_point2):
    # create a new mask with the farthest points
    ellipse_mask = np.zeros_like(mask)
    cv2.line(ellipse_mask, farthest_point1, farthest_point2, 255, thickness=2)

    # fit an ellipse to the farthest points
    contours, _ = cv2.findContours(
        ellipse_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return None
    ellipse = cv2.fitEllipse(contours[0])

    return ellipse








