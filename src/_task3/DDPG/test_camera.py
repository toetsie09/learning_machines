#!/usr/bin/env python2
from __future__ import print_function

import sys
import cv2
import signal
import numpy as np

from robobo_interface import HardwareRobobo


# Green lair
BASE_HSV_MIN = (36, 50, 70)
BASE_HSV_MAX = (86, 255, 255)

# Red food (two sides of hsv)
FOOD_HSV_MIN = [(160, 50, 70), (0, 50, 70)]
FOOD_HSV_MAX = [(180, 240, 240), (10, 240, 240)]


def identify_object(img, min_hsv, max_hsv, min_blob_size=60):
    # Convert to Hue-Saturation-Value (HSV)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Mask out parts of original image that lie in color range
    if type(min_hsv) == tuple:
        mask = cv2.inRange(img, min_hsv, max_hsv)
    else:
        mask0 = cv2.inRange(img, min_hsv[0], max_hsv[0])
        mask1 = cv2.inRange(img, min_hsv[1], max_hsv[1])
        mask = cv2.bitwise_or(mask0, mask1)

    # Try to find blobs
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if not len(contours):
        return mask, np.array([1, 0, 0])  # ('not found', x, y)

    # Select largest contour from the mask
    largest_contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest_contour) < min_blob_size:
        return mask, np.array([1, 0, 0])  # ('not found', x, y)

    # Determine center of the blob
    M = cv2.moments(largest_contour)
    x = int(M["m10"] / M["m00"])
    y = int(M["m01"] / M["m00"])

    # Normalize x and y relative to image centroid and image width/height
    y_norm = 2 * (y / img.shape[0]) - 1
    x_norm = 2 * (x / img.shape[1]) - 1
    return mask, np.array([0, x_norm, y_norm])  # ('found', x, y)


def show_camera(img, food_mask, base_mask, state):
    new_img = np.zeros(img.shape, dtype=np.uint8)
    h, w, _ = new_img.shape

    # Render masks
    new_img[:, :, 0] = np.mean(img, axis=-1)
    new_img[:, :, 1] = base_mask // 2 + new_img[:, :, 0] // 2
    new_img[:, :, 2] = food_mask // 2 + new_img[:, :, 0] // 2

    # Render identification of food
    x0 = int(w * (state[1] + 1) / 2)
    y0 = int(h * (state[2] + 1) / 2)
    new_img[y0 - 8:y0 + 8, x0 - 8:x0 + 8] = (0, 0, 255)

    # Render identification of base
    x1 = int(w * (state[4] + 1) / 2)
    y1 = int(h * (state[5] + 1) / 2)
    new_img[y1 - 8:y1 + 8, x1 - 8:x1 + 8] = (0, 255, 0)

    cv2.imshow('preview', new_img)
    cv2.waitKey(500)


def get_state(robot):
    # Locate Food and Lair objects in camera image
    img = robot.take_picture()

    # Processing!
    img = img[::-1, ::-1]
    img = cv2.medianBlur(img, 11)

    food_mask, food = identify_object(img, FOOD_HSV_MIN, FOOD_HSV_MAX, min_blob_size=8)
    base_mask, base = identify_object(img, BASE_HSV_MIN, BASE_HSV_MAX, min_blob_size=8)

    show_camera(img)

    return np.concatenate([food, base], axis=0)


def test_controller(robot, max_steps):
    """ Test the Robobo camera with identify_object().
    """
    for i in range(max_steps):
        state = get_state(robot)


if __name__ == "__main__":
    # Callback function to save controller on exit
    def terminate(signal_number=None, frame=None):
        print('\nDone!')
        sys.exit(1)
    signal.signal(signal.SIGINT, terminate)

    print('ready')

    # Run controller and print results
    robobo = HardwareRobobo(ip='192.168.1.184')
    test_controller(robobo, max_steps=100)