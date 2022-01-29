#!/usr/bin/env python2
from __future__ import print_function
import sys
import cv2
import signal
import pickle
import numpy as np

from robobo_interface import SimulatedRobobo

# Green base
BASE_HSV_MIN = (36, 50, 70)
BASE_HSV_MAX = (86, 255, 255)

# Red food (two sides of hsv)
FOOD_HSV_MIN = [(160, 50, 70), (0, 50, 70)]
FOOD_HSV_MAX = [(180, 255, 255), (10, 255, 255)]


def identify_object(img, min_hsv, max_hsv, min_blob_size=8):
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
        return np.array([1, 0, 0])  # ('not found', x, y)

    # Select largest contour from the mask
    largest_contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest_contour) < min_blob_size:
        return np.array([1, 0, 0])  # ('not found', x, y)

    # Determine center of the blob
    M = cv2.moments(largest_contour)
    x = int(M["m10"] / M["m00"])
    y = int(M["m01"] / M["m00"])

    # Normalize x and y relative to image centroid and image width/height
    y_norm = 2 * (y / img.shape[0]) - 1
    x_norm = 2 * (x / img.shape[1]) - 1
    return np.array([0, x_norm, y_norm])  # ('found', x, y)


def in_gripper(ir_sensors, thres=0.1):
    value = ir_sensors[3]  # Front-C in reduced sensor array
    if type(value) != bool and value < thres:
        return np.array([1])
    return np.array([0])


def get_state(robot):
    # Locate Food and Lair objects in camera image
    img = robot.take_picture()
    food = identify_object(img, FOOD_HSV_MIN, FOOD_HSV_MAX, min_blob_size=8)
    lair = identify_object(img, BASE_HSV_MIN, BASE_HSV_MAX, min_blob_size=8)

    # Check if some object is in gripper
    obj_in_gripper = in_gripper(robot.get_sensor_state())

    return np.concatenate([food, lair, obj_in_gripper], axis=0)


def to_robobo_commands(action, forward_drive=10, angular_drive=7):
    """ Take an action and converts it into left/right wheel
        commands for the Robobo robot.
    """
    y0, y1 = action
    t = (y0 + 1) / 2
    left_drive = forward_drive * t + angular_drive * y1
    right_drive = forward_drive * t - angular_drive * y1
    return left_drive, right_drive


def test_controller(robot, controller, max_steps, episodes):
    """ Train the Robobo controller in simulation with DDPG.
    """
    evaluation_rewards = []

    for ep in range(episodes):
        rewards = []

        # Start episode!
        robot.start(randomize_arena=True, hide_render=False)

        for _ in range(max_steps):
            # Select action
            state = get_state(robot)
            print(state)
            action = controller.select_action(state)

            # Perform action
            robot.move(*to_robobo_commands(action))

        robot.stop()

        # Save stats accumulated over episode
        evaluation_rewards.append(rewards)

    # Save training stats
    with open('evaluation_rewards.pkl', 'wb') as file:
        pickle.dump(evaluation_rewards, file)


if __name__ == "__main__":
    # Callback function to save controller on exit
    def terminate(signal_number=None, frame=None):
        print('\nDone!')
        sys.exit(1)
    signal.signal(signal.SIGINT, terminate)

    with open('models/Task3_DDPG.pkl', 'rb') as file:
        ddpg_controller = pickle.load(file)

    # optimize controller with DDPG
    robobo = SimulatedRobobo(ip='192.168.1.113', robot_id='')
    test_controller(robobo, ddpg_controller, max_steps=500, episodes=100)
