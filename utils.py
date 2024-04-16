import random

import cv2
import numpy as np
import scipy
from matplotlib import pyplot as plt
from scipy.ndimage import median_filter
import scipy.ndimage.filters as filters


import datetime
import errno
import os
import time
from collections import defaultdict, deque

import torch
import torch.distributed as dist

from scipy import ndimage

def remove_small_disconnected_regions(masks):
    # Convert masks to binary format
    masks_binary = (masks > 0.5).astype(np.uint8)

    # Label connected components in the masks
    labeled_masks, num_labels = ndimage.label(masks_binary)

    # Calculate the size of each connected component
    component_sizes = np.bincount(labeled_masks.flatten())

    # Find the label corresponding to the largest connected component
    largest_label = np.argmax(component_sizes[1:]) + 1

    # Create a mask with only the largest connected component
    largest_mask = labeled_masks == largest_label

    return largest_mask.astype(bool)






class SmoothedValue:
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device="cuda")
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median, avg=self.avg, global_avg=self.global_avg, max=self.max, value=self.value
        )


def all_gather(data):
    """
    Run all_gather on arbitrary picklable data (not necessarily tensors)
    Args:
        data: any picklable object
    Returns:
        list[data]: list of data gathered from each rank
    """
    world_size = get_world_size()
    if world_size == 1:
        return [data]
    data_list = [None] * world_size
    dist.all_gather_object(data_list, data)
    return data_list


def reduce_dict(input_dict, average=True):
    """
    Args:
        input_dict (dict): all the values will be reduced
        average (bool): whether to do average or sum
    Reduce the values in the dictionary from all processes so that all processes
    have the averaged results. Returns a dict with the same fields as
    input_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return input_dict
    with torch.inference_mode():
        names = []
        values = []
        # sort the keys so that they are consistent across processes
        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])
        values = torch.stack(values, dim=0)
        dist.all_reduce(values)
        if average:
            values /= world_size
        reduced_dict = {k: v for k, v in zip(names, values)}
    return reduced_dict


def print_frame_count(video_path):
    cap = cv2.VideoCapture(video_path)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    print(f"Number of frames in the video: {length}")







class MetricLogger:
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{attr}'")

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(f"{name}: {str(meter)}")
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ""
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt="{avg:.4f}")
        data_time = SmoothedValue(fmt="{avg:.4f}")
        space_fmt = ":" + str(len(str(len(iterable)))) + "d"
        if torch.cuda.is_available():
            log_msg = self.delimiter.join(
                [
                    header,
                    "[{0" + space_fmt + "}/{1}]",
                    "eta: {eta}",
                    "{meters}",
                    "time: {time}",
                    "data: {data}",
                    "max mem: {memory:.0f}",
                ]
            )
        else:
            log_msg = self.delimiter.join(
                [header, "[{0" + space_fmt + "}/{1}]", "eta: {eta}", "{meters}", "time: {time}", "data: {data}"]
            )
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(
                        log_msg.format(
                            i,
                            len(iterable),
                            eta=eta_string,
                            meters=str(self),
                            time=str(iter_time),
                            data=str(data_time),
                            memory=torch.cuda.max_memory_allocated() / MB,
                        )
                    )
                else:
                    print(
                        log_msg.format(
                            i, len(iterable), eta=eta_string, meters=str(self), time=str(iter_time), data=str(data_time)
                        )
                    )
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print(f"{header} Total time: {total_time_str} ({total_time / len(iterable):.4f} s / it)")


def collate_fn(batch):
    return tuple(zip(*batch))


def mkdir(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__

    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_mode(args):
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.gpu = int(os.environ["LOCAL_RANK"])
    elif "SLURM_PROCID" in os.environ:
        args.rank = int(os.environ["SLURM_PROCID"])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print("Not using distributed mode")
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = "nccl"
    print(f"| distributed init (rank {args.rank}): {args.dist_url}", flush=True)
    torch.distributed.init_process_group(
        backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size, rank=args.rank
    )
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)




# function for IoU calculation for bbox in xywh format
def bbox_iou(box1, box2):
    iou = 0
    # calculate intersection area
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[0] + box1[2], box2[0] + box2[2])
    y2 = min(box1[1] + box1[3], box2[1] + box2[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    # calculate union area
    union = box1[2] * box1[3] + box2[2] * box2[3] - intersection
    # calculate IoU
    iou = intersection / union
    return iou



def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)
    polygons = []
    color = []
    for ann in sorted_anns:
        m = ann['segmentation']
        img = np.ones((m.shape[0], m.shape[1], 3))
        color_mask = np.random.random((1, 3)).tolist()[0]
        for i in range(3):
            img[:, :, i] = color_mask[i]
        ax.imshow(np.dstack((img, m * 0.35)))




def generate_sam_mask(mask):

    # pad the mask to be square
    h, w = mask.shape
    if h > w:
        pad = (h - w) // 2
        mask = np.pad(mask, ((0, 0), (pad, pad)), 'constant', constant_values=0)
    elif w > h:
        pad = (w - h) // 2
        mask = np.pad(mask, ((pad, pad), (0, 0)), 'constant', constant_values=0)

    # print(mask.shape, "mask shape before resize")
    # resize the mask to 256x256
    mask = cv2.resize(mask, (256, 256))

    # print(mask.shape, "mask shape after resize")

    # make the dimensions 1x256x256
    mask = np.expand_dims(mask, axis=0)

    return mask





def get_color_gradient(displacement, n, cmap='jet'):
    """
    Generate a color based on a specified color gradient for a given displacement.

    Args:
        displacement (float): The displacement value between -n and +n.
        n (float): The maximum displacement value.
        cmap (str, optional): The name of the color map. Default is 'jet'.

    Returns:
        tuple: The BGR color values.
    """
    # Normalize displacement to range [0, 1]
    normalized_displacement = (displacement + n) / (2 * n)

    # Get color from specified color map
    color = plt.cm.get_cmap(cmap)(normalized_displacement)[:3]

    # Scale color values to range [0, 255]
    scaled_color = tuple(np.round(np.array(color) * 255).astype(int))

    # Convert RGB to BGR
    bgr_color = (scaled_color[2], scaled_color[1], scaled_color[0])

    return bgr_color


def get_color_gradient_mask(mask, displacements_array, max_disp, cmap='jet'):
    """
    Generate a colored mask where white pixels are replaced with the corresponding displacement color per row.

    Args:
        mask (np.ndarray): The binary mask.
        displacements_array (np.ndarray): The displacement values for each row.
        max_disp (float): The maximum displacement value.
        cmap (str, optional): The name of the color map. Default is 'jet'.

    Returns:
        np.ndarray: The colored mask.
    """
    height, width = mask.shape
    colored_mask = np.zeros((height, width, 3), dtype=np.uint8)

    # make the displacement positive onyl
    print("displacements_array", displacements_array)


    # print("max_disp", max_disp)


    # apply filter to make the array to smooth it and remove noise with window size 55
    # displacements_array = scipy.ndimage.median_filter(displacements_array, size=5)
    # displacements_array = filters.gaussian_filter1d(displacements_array, sigma=15)


    for row in range(height):
        displacement = displacements_array[row]



        # Normalize displacement to range [0, 1]
        normalized_displacement = displacement/np.max(displacements_array)
        print("normalized_displacement", normalized_displacement)

        # Get color from specified color map
        color = plt.get_cmap(cmap)(normalized_displacement)[:3]

        # print("color", color)
        # Scale color values to range [0, 255]
        bgr_color = (int(color[2]*255), int(color[1]*255), int(color[0]*255))
        # print("bgr_color", bgr_color)


        # Replace white pixels in the row with the displacement color
        colored_mask[row, mask[row, :] == 255, :] = bgr_color


    return colored_mask



# # utils for image processing
# import cv2   # pip install opencv-python
# import numpy as np
# from scipy.ndimage import label
# import matplotlib.pyplot as plt # pip install matplotlib
# import open3d as o3d
#
#
#
# def imshow_mask(img, percent):
#     height, width = img.shape[:2]
#     scale_percent = int(percent*100)  # desired percentage of original size
#
#     width = int(width * scale_percent / 100)
#     height = int(height * scale_percent / 100)
#
#     dim = (width, height)
#
#     resized_img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
#     cv2.imshow("Resized Mask", resized_img)
#     # cv2.waitKey(0)
#     # cv2.destroyAllWindows()
#
# def imshow_img(img, percent):
#     height, width = img.shape[:2]
#     scale_percent = int(percent*100)  # desired percentage of original size
#
#     width = int(width * scale_percent / 100)
#     height = int(height * scale_percent / 100)
#
#     dim = (width, height)
#
#     resized_img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
#     cv2.imshow("Resized Image", resized_img)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
#
#
#
# def largest_submask_fill(mask):
#
#
#     # find the largest connected component in the binary mask
#     labeled_mask, num_labels = label(mask)
#     largest_label = 0
#     largest_size = 0
#     for i in range(1, num_labels + 1):
#         size = np.sum(labeled_mask == i)
#         if size > largest_size:
#             largest_size = size
#             largest_label = i
#
#     # create a binary mask for the largest connected component
#     largest_component = (labeled_mask == largest_label)
#
#     # fill the holes inside the largest connected component
#     result = np.copy(largest_component)
#     result[np.logical_not(mask)] = 0
#
#     return result
#
#
#
# def apply_mask(mask_2, mask_3):
#
#     result = np.copy(mask_2)
#     result[mask_3 == 1] = 0
#
#     return result
#
# def plot3d(stacked_masks):
#     non_zero_indices = stacked_masks.nonzero()
#     z = stacked_masks[non_zero_indices]
#
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')
#     ax.scatter(non_zero_indices[0], non_zero_indices[1], non_zero_indices[2], c='blue', marker='.', s=1, alpha=0.05)
#
#     ax.set_xlabel('Height')
#     ax.set_ylabel('Width')
#     ax.set_zlabel('Mask Number')
#     ax.set_aspect('equal')
#
#     plt.show()
#
#
#
# def save_to_ply(stacked_masks):
#     non_zero_indices = stacked_masks.nonzero()
#     x = non_zero_indices[0]
#     y = non_zero_indices[1]
#     z = non_zero_indices[2]
#
#     points = np.column_stack((x, y, z))
#
#     with open("3d.ply", "w") as f:
#         f.write("ply\n")
#         f.write("format ascii 1.0\n")
#         f.write(f"element vertex {points.shape[0]}\n")
#         f.write("property float x\n")
#         f.write("property float y\n")
#         f.write("property float z\n")
#         f.write("end_header\n")
#
#         for point in points:
#             f.write(f"{point[0]} {point[1]} {point[2]}\n")
#
#
# def show_ply(file_path):
#     pcd = o3d.io.read_point_cloud(file_path)
#     o3d.visualization.draw_geometries([pcd])
#
#
#
# def get_submasks(mask):
#     submasks = []
#     mask_copy = mask.copy()
#
#     # find all connected components in the mask
#     num_labels, labels = cv2.connectedComponents(mask_copy)
#
#     # loop through each connected component and create a submask
#     for label in range(1, num_labels):
#         # create a new mask with only the current connected component
#         submask = np.zeros_like(mask_copy)
#         submask[labels == label] = 1
#
#         submasks.append(submask)
#
#     return submasks
#
#
# def colorize_mask(mask):
#     # Define color map for each label
#     colors = {0: [0, 0, 0],  # black for label 0
#               1: [0, 0, 255],  # red for label 1: GEP
#               2: [0, 255, 0],  # green for label 2: LOF
#               3: [255, 255, 255]}  # yellow for label 3: KH
#     # Create RGB array from mask and color map
#     rgb = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
#     for label in colors:
#         rgb[mask == label] = colors[label]
#
#     return rgb
#
#
#
#
# def find_farthest_points(cnt, mask):
#     # compute pairwise distances between points on the contour
#     dists = cv2.distanceTransform(cv2.bitwise_not(cv2.drawContours(
#         np.zeros_like(mask), [cnt], 0, 255, cv2.FILLED)), cv2.DIST_L2, 5)
#
#     # find the farthest points on the contour
#     max_idx = np.unravel_index(np.argmax(dists), dists.shape)
#     farthest_point1 = tuple(max_idx[::-1])
#     dists[max_idx] = 0
#     max_idx = np.unravel_index(np.argmax(dists), dists.shape)
#     farthest_point2 = tuple(max_idx[::-1])
#
#     return farthest_point1, farthest_point2
#
#
# def fit_ellipse(mask, farthest_point1, farthest_point2):
#     # create a new mask with the farthest points
#     ellipse_mask = np.zeros_like(mask)
#     cv2.line(ellipse_mask, farthest_point1, farthest_point2, 255, thickness=2)
#
#     # fit an ellipse to the farthest points
#     contours, _ = cv2.findContours(
#         ellipse_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#     if len(contours) == 0:
#         return None
#     ellipse = cv2.fitEllipse(contours[0])
#
#     return ellipse
#
#
#
#
#
#
#
#
