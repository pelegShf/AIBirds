import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from mpmath import inf
from ultralytics import YOLO

from consts import SLINGSHOT_BOUNDRIES, CROP_X, CROP_Y, DEFAULT_SLINGSHOT


def get_slingshot(img_dir, model='./vision/best.pt'):
    model = YOLO(model)
    results = model(img_dir)  # predict on an image

    y_min = inf
    x_min = inf
    for result in results:
        for box in result.boxes:
            # Uncomment lines bellow to see YOLO rectangles
            # x1, y1, x2, y2 = box.xyxy[0]
            # cv2.rectangle(cropped_image_rgb, (int(x1.item()), int(y1.item())), (int(x2.item()), int(y2.item())), (0, 255, 0), 2)

            if box.cls[0] == 8.:
                dim = box.xyxy[0]
                x_center = ((dim[0] + dim[2]) / 2).item()
                y_center = ((dim[1] + dim[3]) / 2).item()
                if SLINGSHOT_BOUNDRIES[0] < x_center < SLINGSHOT_BOUNDRIES[2] and SLINGSHOT_BOUNDRIES[1] < y_center < \
                        SLINGSHOT_BOUNDRIES[3]:
                    if y_center < y_min:
                        x_min = x_center
                        y_min = y_center
        # plt.imshow(cropped_image_rgb)
        # plt.show()
        if x_min == inf or y_min == inf:
            x_min = DEFAULT_SLINGSHOT[0] - CROP_X
            y_min = DEFAULT_SLINGSHOT[1] - CROP_Y

    return int(x_min) + CROP_X, int(y_min - 24) + CROP_Y


def crop_img(img_dir):
    image = Image.open(img_dir)
    img = np.asarray(image)
    crop = img[110:400, 70:800]
    cropped_image_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

    cropped_image_path = 'tmp.jpg'
    cv2.imwrite(cropped_image_path, cropped_image_rgb)
    return cropped_image_rgb, cropped_image_path
