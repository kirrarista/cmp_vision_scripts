import sys
import cv2
import os
import math
from os import listdir
import numpy as np
from tqdm.auto import tqdm

# Returns the absolute difference between two neighborhoods
def get_diff_abs(neigh_left, neigh_right):
    diff = np.sum(np.abs(neigh_left - neigh_right))
    return diff

# Returns the square difference between two neighborhoods
def get_diff_sq(neigh_left, neigh_right):
    diff = np.sum(np.square(neigh_left - neigh_right))
    return diff

# Finds the best matching neighborhood in the stripe
def find_best_match(neigh_left, stripe, max_disp):

    min_diff = np.inf
    best_match = 0

    for i in range(0, stripe.shape[1] - neigh_left.shape[1] + 1):

        neigh_right = stripe[:, i:i + neigh_left.shape[1]]

        # diff = get_diff_abs(neigh_left, neigh_right)
        diff = get_diff_sq(neigh_left, neigh_right)

        if diff < min_diff:
            min_diff = diff
            best_match = i
    
    return best_match


if __name__ == "__main__":

    # Check command line arguments
    if len(sys.argv) != 5:
        print("Usage: python3 prog2.py <left_input_img> <right_input_img> <scale_factor> <output_img>")
        sys.exit(1)

    # Parse command line arguments
    in_left_path = sys.argv[1]
    in_right_path = sys.argv[2]
    scale_factor = float(sys.argv[3])
    out_path = sys.argv[4]

    print("Input left image: " + in_left_path)
    print("Input right image: " + in_right_path)
    print("Scale factor: " + str(scale_factor))
    print("Output image: " + out_path)

    # Read input images
    left = cv2.imread(in_left_path)
    right = cv2.imread(in_right_path)

    max_disp = math.ceil(256 / scale_factor)
    
    window_size = 9 # Change this to 3, 5, 7, 9, 11, 13, 15

    num_zeroes = window_size // 2

    # Pad images with zeroes
    padded_left = np.pad(left, ((num_zeroes, num_zeroes), (num_zeroes, num_zeroes), (0, 0)), 'constant')
    padded_right = np.pad(right, ((num_zeroes, num_zeroes), (num_zeroes, num_zeroes), (0, 0)), 'constant')

    top_left = (num_zeroes, num_zeroes)
    bottom_right = (left.shape[0] + num_zeroes, left.shape[1] + num_zeroes)

    # Create disparity map
    disparity = np.zeros((left.shape[0], left.shape[1]))


    # Calculate disparity values
    for i in tqdm(range(top_left[0], bottom_right[0])):
        for j in range(top_left[1], bottom_right[1]):
            
            stripe_left_index = max(0, j - num_zeroes - max_disp)
            stripe_right_index = min(j + num_zeroes + 1, padded_right.shape[1])

            stripe = padded_right[i - num_zeroes:i + num_zeroes + 1, stripe_left_index:stripe_right_index]
            neigh_left = padded_left[i - num_zeroes:i + num_zeroes + 1, j - num_zeroes:j + num_zeroes + 1]

            right_pix = find_best_match(neigh_left, stripe, max_disp)

            disp = j - (stripe_left_index + right_pix)

            scaled_disp = disp * scale_factor

            disparity[i - num_zeroes, j - num_zeroes] = np.uint(scaled_disp)
    
    # Normalize disparity map
    cv2.normalize(disparity, disparity, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    # Write output image
    cv2.imwrite(out_path, disparity)
