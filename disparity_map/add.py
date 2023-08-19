# Helper file for match_disparity.py

import sys
import cv2
import os
import math
from os import listdir
import numpy as np
from tqdm.auto import tqdm

# Read input images
im1 = cv2.imread("results_art/3.png")
im2 = cv2.imread("results_art/9.png")

# Add the images together and get the average
im3 = cv2.addWeighted(im1, 0.5, im2, 0.5, 0)

# Save the image
cv2.imwrite("results_art/avg.png", im3)