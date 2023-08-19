import sys
import cv2
import os
import math
from os import listdir
import matplotlib.pyplot as plt
import numpy as np

# Check command line arguments
if len(sys.argv) != 3:
    print("Usage: python3 validate.py <my_dir> <ground_truth>")
    sys.exit(1)

# Parse command line arguments
my_dir = sys.argv[1]
ground_truth = sys.argv[2]

# Read in the images from my_dir
arr = []
for img in os.listdir(my_dir):
    arr.append((cv2.imread(my_dir + "/" + img), img))

# Sort the images by name
arr.sort(key=lambda x: int(x[1][:-4]))

# Read in the ground truth
gt = cv2.imread(ground_truth)

# Compare the images
diff = []
square_diff = []
for mine in arr:
    # Find percent difference
    diff.append(int((np.sum(np.abs(mine[0] - gt)) / (gt.shape[0] * gt.shape[1] * gt.shape[2]))))
    square_diff.append(int((np.sum(np.square(mine[0] - gt)) / (gt.shape[0] * gt.shape[1] * gt.shape[2]))))

output = []
for i in range(len(diff)):
    output.append((arr[i][1], diff[i], square_diff[i]))
    print(arr[i][1], "abs_diff:", diff[i], "sqr_diff:", square_diff[i])

# Find the best image
best_abs = min(output, key=lambda x: x[1])
best_sqr = min(output, key=lambda x: x[2])

print()
print("Best absolute difference:", best_abs[0])
print("Best square difference:", best_sqr[0])

# Plot the results as a bar chart
plt.bar([x[0] for x in output], [x[1] for x in output])
plt.title("Absolute Difference")
plt.xlabel("Image")
plt.ylabel("Difference")
plt.savefig("stats/cones_stats/abs_diff.png")
# Start a new plot
plt.clf()
plt.bar([x[0] for x in output], [x[2] for x in output])
plt.title("Square Difference")
plt.xlabel("Image")
plt.ylabel("Difference")
plt.savefig("stats/cones_stats/sqr_diff.png")