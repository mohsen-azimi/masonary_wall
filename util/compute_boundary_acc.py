import numpy as np
import cv2
from PIL import Image
from matplotlib import pyplot as plt


def get_disk_kernel(radius):
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius*2+1, radius*2+1))

def compute_boundary_acc(gt, seg, mask):

    gt = gt.astype(np.uint8)
    seg = seg.astype(np.uint8)
    mask = mask.astype(np.uint8)

    h, w = gt.shape

    min_radius = 1
    max_radius = (w+h)/150
    num_steps = 10

    seg_acc = [None] * num_steps
    mask_acc = [None] * num_steps

    for i in range(num_steps):
        curr_radius = min_radius + int((max_radius-min_radius)/num_steps*i)
        # print(f"curr_radius: {curr_radius}, min_radius: {min_radius}, max_radius: {max_radius}, num_steps: {num_steps}")

        kernel = get_disk_kernel(curr_radius)
        # show the kernel
        # save SVG of the kernel
        # kernel_uint8 = kernel.astype(np.uint8)*255
        # plt.imshow(kernel_uint8, cmap='gray')
        # plt.axis('off')
        # plt.savefig(f"C:/Users\Mohsen\Dropbox/01_SAM++\Word\Figures\kernels/kernel_{curr_radius}.svg", format='svg', dpi=300)






        boundary_region = cv2.morphologyEx(gt, cv2.MORPH_GRADIENT, kernel) > 0

        # print("boundary_region", boundary_region.shape, boundary_region.dtype, boundary_region.min(), boundary_region.max())
        # # show the boundary region
        # cv2.imshow("boundary_region", boundary_region.astype(np.uint8)*255)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        # # Display and save the image
        # fig = plt.figure()
        # # plt.imshow(kernel_uint8, cmap='gray') # other cmaps: 'gray', 'jet', 'viridis'
        # # save a cropped version of the boundary region
        # crop_top = 350
        # crop_bottom = 680
        # crop_left = 450
        # crop_right = 570
        # new_boundary_region = boundary_region[crop_top:crop_bottom, crop_left:crop_right]
        # plt.imshow(new_boundary_region, cmap='gray')
        # plt.axis('off')
        # # plt.show()
        # plt.savefig(f"C:/Users\Mohsen\Dropbox/01_SAM++\Word\Figures\kernels/gradient_{curr_radius}.svg", format='svg', dpi=300)


        gt_in_bound = gt[boundary_region]
        seg_in_bound = seg[boundary_region]
        mask_in_bound = mask[boundary_region]

        num_edge_pixels = (boundary_region).sum()
        num_seg_gd_pix = ((gt_in_bound) * (seg_in_bound) + (1-gt_in_bound) * (1-seg_in_bound)).sum()
        num_mask_gd_pix = ((gt_in_bound) * (mask_in_bound) + (1-gt_in_bound) * (1-mask_in_bound)).sum()

        seg_acc[i] = num_seg_gd_pix / num_edge_pixels
        mask_acc[i] = num_mask_gd_pix / num_edge_pixels

    return sum(seg_acc)/num_steps, sum(mask_acc)/num_steps

def compute_boundary_acc_multi_class(gt, seg, mask):
    h, w = gt.shape

    min_radius = 1
    max_radius = (w+h)/300
    num_steps = 5

    seg_acc = [None] * num_steps
    mask_acc = [None] * num_steps

    classes = np.unique(gt)

    for i in range(num_steps):
        curr_radius = min_radius + int((max_radius-min_radius)/num_steps*i)

        kernel = get_disk_kernel(curr_radius)

        boundary_region = np.zeros_like(gt)
        for c in classes:
            # Skip void
            if c == 0:
                continue

            gt_class = (gt == c).astype(np.uint8)
            class_bound = cv2.morphologyEx(gt_class, cv2.MORPH_GRADIENT, kernel)
            boundary_region += class_bound

        boundary_region = boundary_region > 0

        gt_in_bound = gt[boundary_region]
        seg_in_bound = seg[boundary_region]
        mask_in_bound = mask[boundary_region]

        void_count = (gt_in_bound == 0).sum()

        num_edge_pixels = (boundary_region).sum()
        num_seg_gd_pix = (gt_in_bound == seg_in_bound).sum()
        num_mask_gd_pix = (gt_in_bound == mask_in_bound).sum()

        seg_acc[i] = (num_seg_gd_pix-void_count) / (num_edge_pixels-void_count)
        mask_acc[i] = (num_mask_gd_pix-void_count) / (num_edge_pixels-void_count)

    return sum(seg_acc)/num_steps, sum(mask_acc)/num_steps