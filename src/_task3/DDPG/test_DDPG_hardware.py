#!/usr/bin/env python2
from __future__ import print_function

import sys
import cv2
import signal
import pickle
import numpy as np

from robobo_interface import HardwareRobobo


# Green lair
BASE_HSV_MIN = (36, 70, 100)
BASE_HSV_MAX = (86, 240, 240)

# Red food (two sides of hsv)
FOOD_HSV_MIN = [(160, 70, 100), (0, 70, 100)]
FOOD_HSV_MAX = [(180, 240, 240), (10, 240, 240)]


def identify_object(img, min_hsv, max_hsv, min_blob_size=64):
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
    h, w, _ = img.shape

    # Render identification of food
    if state[0] == 0:
        x0 = int(w * (state[1] + 1) / 2)
        y0 = int(h * (state[2] + 1) / 2)
        img[y0 - 8:y0 + 8, x0 - 8:x0 + 8] = (0, 0, 255)

    # Render identification of base
    if state[3] == 0:
        x1 = int(w * (state[4] + 1) / 2)
        y1 = int(h * (state[5] + 1) / 2)
        img[y1 - 8:y1 + 8, x1 - 8:x1 + 8] = (0, 255, 0)

    cv2.imshow('preview', img)
    cv2.waitKey(200)


def get_state(robot):
    # Locate Food and Lair objects in camera image
    img = robot.take_picture()

    # Processing!
    img = img[::-1, ::-1]
    img = cv2.resize(img, (256, 256))
    img = cv2.medianBlur(img, 5)

    # Identify state
    food_mask, food = identify_object(img, FOOD_HSV_MIN, FOOD_HSV_MAX, min_blob_size=8)
    base_mask, base = identify_object(img, BASE_HSV_MIN, BASE_HSV_MAX, min_blob_size=8)
    state = np.concatenate([food, base], axis=0)

    # Preview
    show_camera(img, food_mask, base_mask, state)
    return state


def to_robobo_commands(action, forward_drive=8, angular_drive=5):
    """ Take an action and converts it into left/right wheel
        commands for the Robobo robot.
    """
    y0, y1 = action
    t = (y0 + 1) / 2
    left_drive = forward_drive * t + angular_drive * y1
    right_drive = forward_drive * t - angular_drive * y1
    return [left_drive, right_drive]


def test_controller(robot, controller, max_steps, episodes):
    """ Test the Robobo controller in hardware with DDPG.
    """
    for ep in range(episodes):
        for _ in range(max_steps):
            # observe new state of the world
            state = get_state(robot)

            # Select action (decrease noise over time)
            print('state', state)
            action = controller.select_action(state)
            print('action', action)

            if state[1] > 0:  # Correct left motor deficit
                action[1] = action[1] + 0.1

            # Perform action
            command = to_robobo_commands(action)
            command[0] = command[0] * 1.5               # Correct left motor deficit
            command = [int(round(x)) for x in command]  # Hardware requires integer input
            robot.move(*command, millis=1000)            # Calibrate duration: .8 * original duration


if __name__ == "__main__":
    # Callback function to save controller on exit
    def terminate(signal_number=None, frame=None):
        print('\nDone!')
        sys.exit(1)
    signal.signal(signal.SIGINT, terminate)

    # load controller from checkpoint
    with open('models/Task3_DDPG_flat_base.pkl', 'rb') as file:
        ddpg_controller = pickle.load(file)

    # Run controller and print results
    robobo = HardwareRobobo(ip='192.168.1.184')
    test_controller(robobo, ddpg_controller, max_steps=100, episodes=100)
