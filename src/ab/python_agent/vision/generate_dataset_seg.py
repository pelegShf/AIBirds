import glob
import os
import random
import shutil
from collections import defaultdict

import cv2
import numpy as np


def get_bbox(mask):
    # mask_ = mask.copy()[:, :, :3]
    # mask_ = mask_[:, :, 0] + mask_[:, :, 1] + mask_[:, :, 2]
    #
    # mask_ = cv2.GaussianBlur(mask_, (3, 3), 0)
    #
    # _, mask_ = cv2.threshold(mask_, 1, 255, cv2.THRESH_BINARY)
    #
    # contours, _ = cv2.findContours(mask_.copy(), 1, 1)  # not copying here will throw an error
    # cv2.drawContours(mask_, contours, -1, (0, 255, 0), 3)
    # cv2.imshow('Contours', mask_)
    # cv2.waitKey(0)
    # rect = cv2.minAreaRect(contours[0])  # basically you can feed this rect into your classifier
    # print(f'rect: {rect}')
    # (x, y), (w, h), a = rect  # a - angle

    H, W = mask.shape[0], mask.shape[1]
    X, Y = H // 2, W // 2
    return int(X), int(Y), int(W) + 4, int(H) + 4


def get_bounding_rect(image):
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Threshold the grayscale image to get a binary image
    _, binary_image = cv2.threshold(gray_image, 1, 255, cv2.THRESH_BINARY)

    # Find contours in the binary image with hierarchical information
    contours, _ = cv2.findContours(binary_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Get the bounding rectangles for all contours
    bounding_rects = [cv2.boundingRect(contour) for contour in contours]

    # Merge all bounding rectangles to get the enclosing rectangle
    x, y, w, h = cv2.boundingRect(cv2.convexHull(contours[0]))

    for rect in bounding_rects:
        x, y, w, h = min(x, rect[0]), min(y, rect[1]), max(w, rect[0] + rect[2]) - x, max(h, rect[1] + rect[3]) - y

    return x, y, w, h


def overlay_img(background, overlay, location):
    x, y = location

    background_width = background.shape[1]
    background_height = background.shape[0]

    if x > background_width or y > background_height:
        return background

    h, w = overlay.shape[0], overlay.shape[1]

    if x + w > background_width:
        w = background_width - x
        overlay = overlay[:, :w]

    if y + h > background_height:
        h = background_height - y
        overlay = overlay[:h]

    if overlay.shape[2] < 4:
        overlay = np.concatenate(
            [
                overlay,
                np.ones((overlay.shape[0], overlay.shape[1], 1), dtype=overlay.dtype) * 255
            ],
            axis=2,
        )

    overlay_image = overlay[..., :]
    mask = overlay[..., 3:] / 255.0

    background[y:y + h, x:x + w] = (1.0 - mask) * background[y:y + h, x:x + w] + mask * overlay_image

    return background


def rotate_object(mat, angle):
    """
    Rotates an image (angle in degrees) and expands image to avoid cropping
    """

    height, width = mat.shape[:2]  # image shape has 3 dimensions
    image_center = (
    width / 2, height / 2)  # getRotationMatrix2D needs coordinates in reverse order (width, height) compared to shape

    rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1.)

    # rotation calculates the cos and sin, taking absolutes of those.
    abs_cos = abs(rotation_mat[0, 0])
    abs_sin = abs(rotation_mat[0, 1])

    # find the new width and height bounds
    bound_w = int(height * abs_sin + width * abs_cos)
    bound_h = int(height * abs_cos + width * abs_sin)

    # subtract old image center (bringing image back to origo) and adding the new image center coordinates
    rotation_mat[0, 2] += bound_w / 2 - image_center[0]
    rotation_mat[1, 2] += bound_h / 2 - image_center[1]

    # rotate image with the new bounds and translated rotation matrix
    rotated_mat = cv2.warpAffine(mat, rotation_mat, (bound_w, bound_h))
    return rotated_mat


def get_label_map(object_dirs: list):
    label_to_images = defaultdict(list)
    label_to_masks = defaultdict(list)
    for object_dir in object_dirs:
        for object_image_path in glob.glob(f"{object_dir}*"):
            if "mask" in object_image_path:
                continue
            object_split = os.path.basename(object_image_path).split("_")
            label = "_".join(object_split[: -1])
            object_num = object_split[-1][:-4]
            mask_path = f"{object_dir}{label}_mask_{object_num}.png"
            label_to_images[label].append(object_image_path)
            label_to_masks[label].append(mask_path)
    return label_to_images, label_to_masks

background_dir = 'vision/resources/angry_birds_backgrounds'
output_dir = 'vision/dataset'
images_dir = os.path.join(output_dir, 'images')
labels_dir = os.path.join(output_dir, 'labels')
train_images_dir = os.path.join(images_dir, 'train')
train_labels_dir = os.path.join(labels_dir, 'train')
val_images_dir = os.path.join(images_dir, 'val')
val_labels_dir = os.path.join(labels_dir, 'val')

for dir_ in [output_dir, images_dir, labels_dir, train_images_dir, train_labels_dir, val_images_dir, val_labels_dir]:
    if os.path.exists(dir_):
        shutil.rmtree(dir_)
    os.makedirs(dir_)

pigs_dir = "/home/david/AIBirds/src/ab/python_agent/vision/resources/pigs/"
birds_dir = "/home/david/AIBirds/src/ab/python_agent/vision/resources/birds/"
ice_dir = "/home/david/AIBirds/src/ab/python_agent/vision/resources/ice/"
stone_dir = "/home/david/AIBirds/src/ab/python_agent/vision/resources/stone/"
wood_dir = "/home/david/AIBirds/src/ab/python_agent/vision/resources/wood/"
tnt_dir = "/home/david/AIBirds/src/ab/python_agent/vision/resources/tnt/"
hill_dir = "/home/david/AIBirds/src/ab/python_agent/vision/resources/hill/"

object_dirs = [pigs_dir, birds_dir, ice_dir, stone_dir, wood_dir, tnt_dir, hill_dir]

label_to_image, label_to_mask = get_label_map(object_dirs)

train_dataset_size = 5000
val_dataset_size = 500

max_object_count = 2


def get_background(background_dir):
    backgrounds_list = os.listdir(background_dir)
    backgrounds_idx = random.randint(0, len(backgrounds_list) - 1)
    backgrounds_file = backgrounds_list[backgrounds_idx]
    background_path = os.path.join(background_dir, backgrounds_file)
    background_ = cv2.imread(background_path)
    background = np.ones((background_.shape[0], background_.shape[1], 4), dtype=np.uint8) * 255
    background[:, :, :3] = background_
    return background


def get_objects(max_object_count: int, label_to_image: dict):
    label_to_chosen = defaultdict(list)
    for label, images in label_to_image.items():
        num_objects = random.randint(0, max_object_count)
        label_to_chosen[label] = random.choices(images, k=num_objects)
    return label_to_chosen


def place_object(obj_img: np.ndarray, obj_mask: np.ndarray, background_img: np.ndarray, taken_pixels: list):
    object_height, object_width, _ = obj_img.shape
    background_height, background_width, _ = background_img.shape
    done = False
    while not done:
        try:
            location_y, location_x = random.randint(object_height, background_height - object_height), \
                random.randint(object_width, background_width - object_width)
        except:
            continue

        taken_y, taken_x = np.where(obj_mask != 0)
        taken_x += location_x
        taken_y += location_y
        taken_x_y = []
        done = True
        for idx, x in enumerate(taken_x):
            y = taken_y[idx]
            x_y = f"{x}_{y}"
            if x_y in taken_pixels:
                done = False
                break
            taken_x_y.append(x_y)

    taken_pixels.extend(taken_x_y)
    img_ = overlay_img(background_img, obj_img, (location_x, location_y))
    return img_, taken_x.astype(np.float64), taken_y.astype(np.float64)

def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation = inter)

    # return the resized image
    return resized


labels = list(label_to_image.keys())
print(f"labels: {labels}")

for (dataset_size, images_dir, labels_dir) in [(train_dataset_size, train_images_dir, train_labels_dir),
                                               (val_dataset_size, val_images_dir, val_labels_dir)]:
    for img_idx in range(dataset_size):
        taken_pixels = []
        background_img = get_background(background_dir)
        label_to_chosen = get_objects(max_object_count, label_to_image)
        with open(os.path.join(labels_dir, f"{img_idx}.txt"), "w") as label_file:
            for label_str, obj_imgs in label_to_chosen.items():
                label_int = labels.index(label_str)
                for obj_img_path in obj_imgs:
                    i = label_to_image[label_str].index(obj_img_path)
                    obj_mask_path = label_to_mask[label_str][i]
                    print(obj_img_path)
                    print(obj_mask_path)
                    obj_mask = cv2.imread(obj_mask_path, -1)
                    obj_img = cv2.imread(obj_img_path, -1)

                    h, w, _ = obj_img.shape
                    ratio = h / w
                    resize_h = 20
                    resize_w = 20
                    print(ratio)
                    if ratio < 0.15:
                        resize_w = 60
                        resize_h = 15
                    elif ratio < 0.5:
                        resize_w = 40
                        resize_h = 40
                    if "hill" in label_str:
                        resize_w = random.randint(50, 150)
                        resize_h = random.randint(50, 150)

                    obj_img = image_resize(obj_img, resize_w, resize_h)
                    obj_mask = image_resize(obj_mask, resize_w, resize_h)

                    object_rotation_angle = random.randint(0, 359)
                    obj_img = rotate_object(obj_img, object_rotation_angle)
                    obj_mask = rotate_object(obj_mask, object_rotation_angle)

                    img_, taken_x, taken_y = place_object(obj_img, obj_mask, background_img, taken_pixels)

                    cv2.imwrite(os.path.join(images_dir, f'{img_idx}.jpg'), img_)
                    img_height, img_width, _ = img_.shape
                    taken_x /= img_width
                    taken_y /= img_height
                    line = f"{label_int} "
                    for idx, x in enumerate(taken_x):
                        y = taken_y[idx]
                        line += f"{x} {y} "

                    label_file.write(f"{line[:-1]}\n")
