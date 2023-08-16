import os
import subprocess

import cv2
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

import numpy as np
from PIL import Image
from mpmath import inf
from ultralytics import YOLO

from consts import SLINGSHOT_BOUNDRIES, CROP_X, CROP_Y, DEFAULT_SLINGSHOT, SCREEN_WIDTH, SCREEN_HEIGHT, OFFSETX, OFFSETY


def generate_state(img_dir, img_dir_zoomed, model='./vision/best.pt'):
    # Directory containing the JAR file
    # Navigate to the JAR directory
    # os.chdir(jar_directory)
    jar_file = "./vision/vision.jar"
    main_class = "ab.agentvision.davidAgent"  # Replace with the actual main class
    image_directory = "/screenshot.png"  # Replace with the actual image directory
    zoom_image_directory = "/zoomed_screenshot.png"  # Replace with the actual image directory

    subprocess.run(["java", "-cp", jar_file, main_class, image_directory, zoom_image_directory])
    sling_found, sling, state = create_state('./filename.txt')
    print(f'first sling: {sling["x"]} {sling["y"]} {sling["w"]} {sling["h"]}')
    print(sling_found)
    if not sling_found:
        x, y = find_sling_yolo(img_dir, model)
        return state, x, y
    else:
        return state, (sling["x"] + sling["w"] * OFFSETX), (sling["y"] + sling["w"] * OFFSETX)


def find_sling_yolo(img_dir, model='./vision/best.pt'):
    cropped_img, cropped_img_dir = crop_img(img_dir)
    model = YOLO(model)
    results = model(cropped_img_dir)  # predict on an image

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
    plt.imshow(img)
    crop = img[110:400, 70:800]
    cropped_image_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

    cropped_image_path = 'tmp.png'
    cv2.imwrite(cropped_image_path, cropped_image_rgb)
    return cropped_image_rgb, cropped_image_path


def create_state(file_path):
    yolov8_bbox_data = []
    with open(file_path, 'r') as file:
        lines = file.readlines()
    slingshot_found = False
    sling = {
        'x': 0,
        'y': 0,
        'w': 0,
        'h': 0,
    }
    slingshot = lines[0].strip().split()
    if slingshot[0] == "slingshot":
        slingshot_found = True
        sling['x'] = int(slingshot[1])
        sling['y'] = int(slingshot[2])
        sling['w'] = int(slingshot[3])
        sling['h'] = int(slingshot[4])

    for line in lines[0:-1]:  # Exclude the last line
        values = line.strip().split()
        bbox = {
            "class": values[0],
            "x": int(values[1]),
            "y": int(values[2]),
            "width": int(values[3]),
            "height": int(values[4]),
        }
        yolov8_bbox_data.append(bbox)

    # Create a dictionary to map class labels to unique values
    class_to_value = {
        "unknown": 0,
        "hill": 1,
        "Wood": 2,
        "Ice": 3,
        "Stone": 4,
        "slingshot": 5,
        "pig": 6,
        "tnt": 7,
        # Add more class labels and values as needed
    }
    class_colors = {
        "unknown": "#7A7A7A",
        "Wood": "#9C5E20",
        "Ice": "#EBE2D8",
        "Stone": "#18189E",
        "slingshot": "#9E9818",
        "pig": "#30C618",
        "hill": "#76801E",
        "tnt": "#9E1887",
        # Add more class labels and values as needed
    }
    # Set the image dimensions (replace with your actual image dimensions)
    image_width = SCREEN_WIDTH
    image_height = SCREEN_HEIGHT

    # Create an empty numerical array
    num_cells_x = SCREEN_WIDTH
    num_cells_y = SCREEN_HEIGHT
    numerical_array = np.zeros((num_cells_y, num_cells_x))

    # Convert state array to a numerical array with class values
    for bbox in yolov8_bbox_data:
        x_center = bbox["x"] + bbox["width"] / 2
        y_center = bbox["y"] + bbox["height"] / 2
        cell_x = int(x_center * num_cells_x / image_width)
        cell_y = int(y_center * num_cells_y / image_height)

        class_label = bbox["class"]
        class_value = class_to_value.get(class_label, 0)  # 0 for unknown class

        x_range = int(bbox["width"] * num_cells_x / image_width)
        y_range = int(bbox["height"] * num_cells_y / image_height)

        for y_offset in range(-y_range // 2, y_range // 2):
            for x_offset in range(-x_range // 2, x_range // 2):
                y_idx = min(max(cell_y + y_offset, 0), num_cells_y - 1)
                x_idx = min(max(cell_x + x_offset, 0), num_cells_x - 1)
                numerical_array[y_idx][x_idx] = class_value
    #     VISUAL STUFF - uncomment for human
    # num_unique_classes = len(set(class_to_value.values()))
    # colors = [class_colors.get(label, 'black') for label in class_to_value.keys()]
    # cmap = mcolors.ListedColormap(colors)
    #
    # # Create a color-coded heatmap visualization
    # plt.imshow(numerical_array, cmap=cmap, interpolation='nearest', vmin=0, vmax=num_unique_classes)
    # cbar = plt.colorbar(ticks=np.arange(num_unique_classes + 1))
    # cbar.set_label('Class Value')
    # cbar.ax.set_yticklabels(['Unknown'] + [label for label in class_to_value.keys()])

    # plt.title('State Array Visualization')
    # plt.xlabel('X Cells')
    # plt.ylabel('Y Cells')
    # plt.show()
    return slingshot_found, sling, numerical_array
