import cv2
import numpy as np
import os
import shutil

name = "wood"
save_dir = f"/home/david/AIBirds/src/ab/python_agent/vision/resources/{name}/"
img = cv2.imread(f"/home/david/AIBirds/src/ab/python_agent/vision/resources/{name}.png", cv2.IMREAD_UNCHANGED)
shutil.rmtree(save_dir)
os.makedirs(save_dir)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = 255 - gray

ret, blob = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

contours, hier = cv2.findContours(blob, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)


# Get blob indices with respect to hierarchy
blob_idx = np.squeeze(np.where(hier[0, :, 3] == -1))
# Initialize blob images
blob_imgs = []
# Iterate all blobs
for obj_idx, b_idx in enumerate(np.nditer(blob_idx)):

    # Add outer contour of blob to list
    blob_cnts = [contours[b_idx]]

    tlx, tly, width, height = cv2.boundingRect(blob_cnts[0])
    crop = img[tly: tly + height, tlx: tlx + width, ...]
    mask = blob[tly: tly + height, tlx: tlx + width]
    cv2.imwrite(f"{save_dir}/{name}_{obj_idx}.png", crop)
    cv2.imwrite(f"{save_dir}/{name}_mask_{obj_idx}.png", mask)

#     # Add inner contours of blob to list, if present
#     cnt_idx = np.squeeze(np.where(hier[0, :, 3] == b_idx))
#     if (cnt_idx.size > 0):
#         blob_cnts.extend([contours[c_idx] for c_idx in np.nditer(cnt_idx)])
#
#     # Generate blank BGR image with same size as input; draw contours
#     res = np.zeros((blob.shape[0], blob.shape[1], 3), np.uint8)
#     cv2.drawContours(res, blob_cnts, -1, colors[k % 3], 2)
#     blob_imgs.append(res)
#     k += 1
#
# # Just for visualization: Iterate all blob images
# k = 0
# for res in blob_imgs:
#     cv2.imshow(str(k), res)
#     k += 1
# cv2.waitKey(0)
# cv2.destroyAllWindows()