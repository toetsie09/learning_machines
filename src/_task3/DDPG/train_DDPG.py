#!/usr/bin/env python2
from __future__ import print_function
import sys
import cv2
import signal
import pickle
import numpy as np
from tqdm import tqdm

from robobo_interface import SimulatedRobobo
from DDPG import DDPGAgent

# Green lair
BASE_HSV_MIN = (36, 50, 70)
BASE_HSV_MAX = (86, 255, 255)

# Red food (two sides of hsv)
FOOD_HSV_MIN = [(160, 50, 70), (0, 50, 70)]
FOOD_HSV_MAX = [(180, 255, 255), (10, 255, 255)]


def identify_object(img, min_hsv, max_hsv, min_blob_size=4):
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


def get_state(robot):
    # Locate Food and Lair objects in camera image
    img = robot.take_picture()
    food = identify_object(img, FOOD_HSV_MIN, FOOD_HSV_MAX, min_blob_size=8)
    base = identify_object(img, BASE_HSV_MIN, BASE_HSV_MAX, min_blob_size=8)
    return np.concatenate([food, base], axis=0)


def to_robobo_commands(action, forward_drive=8, angular_drive=5):
    """ Take an action and converts it into left/right wheel
        commands for the Robobo robot.
    """
    y0, y1 = action
    t = (y0 + 1) / 2
    left_drive = forward_drive * t + angular_drive * y1
    right_drive = forward_drive * t - angular_drive * y1
    return left_drive, right_drive


def train_controller(robot, controller, max_steps, episodes):
    """ Train the Robobo controller in simulation with DDPG.
    """
    training_rewards = []

    for ep in range(episodes):
        pbar = tqdm(total=max_steps, position=0, desc=str(ep), leave=True)
        rewards = []

        # Start episode!
        robot.start(randomize_arena=True, hide_render=True)

        # Observe current state
        state = get_state(robot)
        dist_robot_to_food = robot.distance_between('Robobo', 'Food')
        dist_food_to_base = robot.distance_between('Food', 'Base')

        for _ in range(max_steps):
            # Select action (decrease noise over time)
            eps = 0.9 - (0.8 * ep / episodes)
            if np.random.random() < eps:
                action = np.random.uniform(-1, 1, 2)
            else:
                action = controller.select_action(state)

            # Add some noise to simulate actuator imprecision
            noisy_action = action + np.random.normal(0, 0.1, 2)

            # Perform action
            robot.move(*to_robobo_commands(noisy_action))

            # observe new state of the world
            new_state = get_state(robot)
            new_dist_robot_to_food = robot.distance_between('Robobo', 'Food')
            new_dist_food_to_base = robot.distance_between('Food', 'Base')

            # Compute reward as decreased distance between Food, Lair and Robobo
            reward = dist_robot_to_food - new_dist_robot_to_food
            reward += 3 * (dist_food_to_base - new_dist_food_to_base)
            rewards.append(reward)

            # learn from reward (during training episodes ofc)
            controller.save_experience(state, action, reward, new_state)
            controller.update()

            state = new_state
            dist_robot_to_food = new_dist_robot_to_food
            dist_food_to_base = new_dist_food_to_base
            pbar.update(1)

            if new_dist_food_to_base < 0.1:
                break

        robot.stop()

        pbar.set_postfix({'total_rewards': sum(rewards)})
        pbar.close()

        # Save stats accumulated over episode
        training_rewards.append(rewards)

    # Save training stats
    with open('training_rewards.pkl', 'wb') as file:
        pickle.dump(training_rewards, file)


if __name__ == "__main__":
    # Init controller
    ddpg_controller = DDPGAgent(layer_shapes=(6, 24, 8, 2), gamma=0.99, actor_lrate=1e-3,
                                critic_lrate=5e-3, replay_size=96)

    # Callback function to save controller on exit
    def save_controller(signal_number=None, frame=None):
        print("\nSaving controller!")
        with open('models/Task3_DDPG_flat_base_noisy.pkl', 'wb') as file:
            pickle.dump(ddpg_controller, file)
        sys.exit(1)
    signal.signal(signal.SIGINT, save_controller)

    # optimize controller with DDPG
    robobo = SimulatedRobobo(ip='192.168.1.113', robot_id='')
    train_controller(robobo, ddpg_controller, max_steps=100, episodes=300)
    save_controller()
