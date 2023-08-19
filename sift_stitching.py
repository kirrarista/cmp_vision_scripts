import sys
import cv2
import os
from os import listdir
import numpy as np

in_path = sys.argv[1]
out_path = sys.argv[2]

print("Input file: " + in_path)
print("Output file: " + out_path)

arr = []
for img in os.listdir(in_path):
    if (img.endswith(".JPG") or img.endswith(".jpg")):
        arr.append((cv2.imread(in_path + "/" + img), img))

try:
    arr = sorted(arr, key=lambda x: int(x[1][:-4]))
except:
    try:
        arr = sorted(arr, key=lambda x: x[1][:-4])
    except:
        arr = sorted(arr, key=lambda x: int(x[1][:-2]))

for i in range(len(arr)):
    if arr[i][1] == "mosaic.JPG" or arr[i][1] == "mosaic.jpg":
            arr.pop(i)
arr = [x[0] for x in arr]

#reverse order
arr = arr[::-1]

# print(arr)

# for i in range(0, 21):
#     arr.append(cv2.imread(in_path + "/" + str(i) + ".JPG"))

stiched_img = arr[len(arr)//2]

# Initialize SIFT
sift = cv2.xfeatures2d.SIFT_create()

# Initialize Brute Force Matcher
bf = cv2.BFMatcher()

def stitch_images(arr, stiched_img, sift, bf, i, right):
    if right:
        new_img = arr[i]
    else:
        new_img = arr[i]
        new_img = cv2.flip(new_img, 1)

    # Use SIFT to find keypoints and descriptors
    stiched_img_kpts, stiched_img_descr = sift.detectAndCompute(stiched_img, None)
    new_img_kpts, new_img_descr = sift.detectAndCompute(new_img, None)

    # Use Brute Force Matcher to find matches
    matches = bf.knnMatch(stiched_img_descr, new_img_descr, k=2)

    # Apply ratio test
    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append([m])

    # # Draw matches
    # img = cv2.drawMatchesKnn(stiched_img, stiched_img_kpts, new_img, new_img_kpts, good, None, flags=2)

    # Find keypoints in both images
    stiched_img_pts = np.float32([stiched_img_kpts[m[0].queryIdx].pt for m in good]).reshape(-1, 1, 2)
    new_img_pts = np.float32([new_img_kpts[m[0].trainIdx].pt for m in good]).reshape(-1, 1, 2)
    
    # Find homography matrix
    h, mask = cv2.findHomography(new_img_pts, stiched_img_pts, cv2.RANSAC)

    # Apply panorama correction
    width = stiched_img.shape[1] + new_img.shape[1]
    height = stiched_img.shape[0] + new_img.shape[0]
    # Warp image
    warped_img = cv2.warpPerspective(new_img, h, (width, height))

    # Stitch images
    warped_img[0:stiched_img.shape[0], 0:stiched_img.shape[1]] = stiched_img

    # Bounding box
    # Greyscale
    warped_img_copy = cv2.cvtColor(warped_img, cv2.COLOR_BGR2GRAY)
    x, y, w, h = cv2.boundingRect(warped_img_copy)
    warped_img = warped_img[y:y+h, x:x+w]
    return warped_img

for i in range(len(arr)//2, len(arr)//2+5):
    stiched_img = stitch_images(arr, stiched_img, sift, bf, i, right=True)
right = stiched_img

stiched_img = cv2.flip(right, 1)
# stiched_img = cv2.flip(arr[5], 1)

for i in reversed(range(len(arr)//2-5, len(arr)//2)):
    stiched_img = stitch_images(arr, stiched_img, sift, bf, i, right=False)
full = stiched_img

# Save image
cv2.imwrite(out_path, cv2.flip(full, 1))
