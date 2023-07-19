import os
import random
import shutil
from matplotlib import pyplot as plt

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

    # return int(X), int(Y), int(W) + 4, int(H) + 4


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


# def rotate(img, alpha):
#     (h, w) = img.shape[:2]
#     (cX, cY) = (w // 2, h // 2)
#
#     M = cv2.getRotationMatrix2D((cX, cY), alpha, 1.0)
#     rotated = cv2.warpAffine(img, M, (w, h))
#
#     return rotated


def rotate(rotateImage, angle):
    # Taking image height and width
    imgHeight, imgWidth = rotateImage.shape[0], rotateImage.shape[1]

    # Computing the centre x,y coordinates
    # of an image
    centreY, centreX = imgHeight // 2, imgWidth // 2

    # Computing 2D rotation Matrix to rotate an image
    rotationMatrix = cv2.getRotationMatrix2D((centreY, centreX), angle, 1.0)

    # Now will take out sin and cos values from rotationMatrix
    # Also used numpy absolute function to make positive value
    cosofRotationMatrix = np.abs(rotationMatrix[0][0])
    sinofRotationMatrix = np.abs(rotationMatrix[0][1])

    # Now will compute new height & width of
    # an image so that we can use it in
    # warpAffine function to prevent cropping of image sides
    newImageHeight = int((imgHeight * sinofRotationMatrix) +
                         (imgWidth * cosofRotationMatrix))
    newImageWidth = int((imgHeight * cosofRotationMatrix) +
                        (imgWidth * sinofRotationMatrix))

    # After computing the new height & width of an image
    # we also need to update the values of rotation matrix
    rotationMatrix[0][2] += (newImageWidth / 2) - centreX
    rotationMatrix[1][2] += (newImageHeight / 2) - centreY

    # Now, we will perform actual image rotation
    rotatingimage = cv2.warpAffine(
        rotateImage, rotationMatrix, (newImageWidth, newImageHeight))

    return rotatingimage


background_dir = 'vision/resources/angry_birds_backgrounds'
pig_img_path = 'vision/resources/pig.png'
red_bird_img_path = 'vision/resources/red_bird.png'
black_bird_img_path = 'vision/resources/black_bird.png'
green_bird_img_path = 'vision/resources/green_bird.png'
red_big_bird_img_path = 'vision/resources/reb_big_bird.png'
white_bird_img_path = 'vision/resources/white_bird.png'
yellow_bird_img_path = 'vision/resources/yellow_bird.png'

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
    os.mkdir(dir_)

pig_img = cv2.imread(pig_img_path, -1)
red_bird_img = cv2.imread(red_bird_img_path, -1)
black_bird_img = cv2.imread(black_bird_img_path, -1)
green_bird_img = cv2.imread(green_bird_img_path, -1)
red_big_bird_img = cv2.imread(red_big_bird_img_path, -1)
white_bird_img = cv2.imread(white_bird_img_path, -1)
yellow_bird_img = cv2.imread(yellow_bird_img_path, -1)

birds = [red_bird_img, black_bird_img, green_bird_img, red_big_bird_img, white_bird_img, yellow_bird_img]

train_dataset_size = 5000
val_dataset_size = 1000

for d, dataset_size in enumerate([train_dataset_size, val_dataset_size]):
    for j in range(dataset_size):
        alpha = random.randint(0, 359)
        background_path = os.path.join(background_dir,
                                       os.listdir(background_dir)[
                                           random.randint(0, len(os.listdir(background_dir)) - 1)])
        background_ = cv2.imread(background_path)
        background_ = cv2.resize(background_, (1200, 650))
        background = np.ones((background_.shape[0], background_.shape[1], 4), dtype=np.uint8) * 255
        background[:, :, :3] = background_
        bird_index = random.randint(0, len(birds) - 1)
        bird = birds[bird_index]
        for i, obj_img in enumerate([pig_img, bird]):

            resize_ = 50 * random.randint(1, 2)

            img_ = cv2.resize(obj_img, (resize_, resize_))
            img_ = rotate(img_, alpha)
            xc, yc, w, h = get_bbox(img_)

            try:
                location_x, location_y = random.randint(img_.shape[1], background.shape[1] - img_.shape[1]), \
                    random.randint(img_.shape[0], background.shape[0] - img_.shape[0])

                img_ = overlay_img(background, img_, (location_x, location_y))

                img_dir = train_images_dir if d == 0 else val_images_dir  # train or validation dataset
                cv2.imwrite(os.path.join(img_dir, '{}.jpg'.format(str(j))), img_)

                H, W, _ = img_.shape
                lbl_dir = train_labels_dir if d == 0 else val_labels_dir  # train or validation dataset
                with open(os.path.join(lbl_dir, '{}.txt'.format(str(j))), 'a') as f:
                    if i == 0:
                        f.write(
                            '0 {} {} {} {}\n'.format(str((xc + location_x) / W), str((yc + location_y) / H), str(w / W),
                                                     str(h / H)))
                    else:
                        f.write('{} {} {} {} {}\n'.format(str(bird_index + 1), str((xc + location_x) / W),
                                                          str((yc + location_y) / H), str(w / W),
                                                          str(h / H)))
            except Exception:
                pass
