import numpy as np


def yolo8_to_numpy_array(yolo8_file_path, resolution=(840, 480)):
    # Initialize the numpy array with zeros
    width, height = resolution
    numpy_array = np.zeros((height, width))

    with open(yolo8_file_path, 'r') as file:
        for line in file:
            # Each line in the file represents a bounding box in YOLO8 format (class, x, y, width, height)
            class_idx, x_center, y_center, box_width, box_height = map(float, line.split())

            # Convert YOLO8 format to pixel coordinates
            x1 = int((x_center - box_width / 2) * width)
            y1 = int((y_center - box_height / 2) * height)
            x2 = int((x_center + box_width / 2) * width)
            y2 = int((y_center + box_height / 2) * height)

            # Set the corresponding cells in the numpy array to the class value
            numpy_array[y1:y2, x1:x2] = class_idx
    return numpy_array


yolo8_file_path = './dataset/labels/train/1.txt'
resolution = (840, 480)
resulting_numpy_array = yolo8_to_numpy_array(yolo8_file_path, resolution)
print(np.sum(resulting_numpy_array))
file = open("file2.txt", "w+")
content = str(resulting_numpy_array)
file.write(content)
file.close()
