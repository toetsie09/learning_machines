#!/usr/bin/env python2
from __future__ import print_function
import sys
import cv2
import signal
import pickle
import numpy as np

from robobo_interface import SimulatedRobobo

FOOD_HSV_MIN = (36, 0, 0)
FOOD_HSV_MAX = (86, 255, 255)


def identify_food(img, min_hsv, max_hsv, min_blob_size=8):
    # Convert to Hue-Saturation-Value (HSV)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Mask out parts of original image that lie in color range
    mask = cv2.inRange(img, min_hsv, max_hsv)

    # Try to find blobs
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if not len(contours):
        return np.array([1, 0, 0])  # ('no food', x, y)

    # Select largest contour from the mask
    largest_contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest_contour) < min_blob_size:
        return np.array([1, 0, 0])  # ('no food', x, y)

    # Determine center of the blob
    M = cv2.moments(largest_contour)
    x = int(M["m10"] / M["m00"])
    y = int(M["m01"] / M["m00"])

    # Normalize x and y relative to image centroid and image width/height
    y_norm = 2 * (y / img.shape[0]) - 1
    x_norm = 2 * (x / img.shape[1]) - 1
    return np.array([0, x_norm, y_norm])  # ('found food', x, y)


def to_robobo_commands(action, forward_drive=10, angular_drive=5):
    """ Take an action and converts it into left/right wheel
        commands for the Robobo robot.
    """
    y0, y1 = action
    t = (y0 + 1) / 2
    left_drive = forward_drive * t + angular_drive * y1   # Changed!
    right_drive = forward_drive * t - angular_drive * y1  # Changed!
    return left_drive, right_drive


def test_controller(robot, controller, max_steps, episodes):
    """ Train the Robobo controller in simulation with DDPG.
    """
    for ep in range(episodes):
        # Start episode!
        robot.start(randomize_food=True, hide_render=False)
        food_collected = 0

        for _ in range(max_steps):
            # observe new state of the world
            state = identify_food(robot.take_picture(), FOOD_HSV_MIN, FOOD_HSV_MAX)

            # Select action (decrease noise over time)
            print('state', state)
            action = controller.select_action(state)
            print('action', action)

            # Perform action
            robot.move(*to_robobo_commands(action))

            # Did we find food or collide with a wall?
            collision, hit_food = robot.has_collided()
            food_collected += int(hit_food)

            # End episode on collision
            if (collision and not hit_food) or food_collected == 8:
                break

        robot.stop()


if __name__ == "__main__":
    # Callback function to save controller on exit
    def terminate(signal_number=None, frame=None):
        print('\nDone!')
        sys.exit(1)
    signal.signal(signal.SIGINT, terminate)

    with open('models/Task2_DDPG_final.pkl', 'rb') as file:
        ddpg_controller = pickle.load(file)

    # optimize controller with DDPG
    robobo = SimulatedRobobo(ip='192.168.1.113', robot_id='')
    test_controller(robobo, ddpg_controller, max_steps=500, episodes=200)
