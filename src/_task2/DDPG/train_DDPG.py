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


FOOD_HSV_MIN = (36, 0, 0)
FOOD_HSV_MAX = (86, 255, 255)


def ir_to_proximity(sim_sensor_dists, d_max=0.2):
    """ Convert distances to proximity values
    """
    values = []
    for d in sim_sensor_dists:
        if type(d) == bool:
            values += [0]
        else:
            values += [max(0, (d_max - d) / d_max)]
    return np.array(values)


def identify_food(img, min_hsv, max_hsv, min_blob_size=8, show=False):
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

    if show:
        # Show preview with x,y pair
        preview = cv2.merge((mask, mask, mask))
        preview[int(y)-2:int(y)+2, int(x)-2:int(x)+2] = (255, 0, 0)
        cv2.imshow('preview', preview)
        cv2.waitKey(100)

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


def compute_reward(collision, hit_food):
    """ Computes the reward associated with the current state of the environment

        +1000  for collision with a food item
        -100   for collision with an obstacle/wall
        -1     otherwise (punish robot for being passive)
    """
    if hit_food:
        return 100
    elif collision:  # (includes the case where there's food but collided at rear)
        return -100
    return -1


def train_controller(robot, controller, max_steps, episodes):
    """ Train the Robobo controller in simulation with DDPG.
    """
    training_rewards = []
    training_collected_foods = []

    for ep in range(episodes):
        pbar = tqdm(total=max_steps, position=0, desc=str(ep), leave=True)
        rewards = []
        food_collected = 0

        # Start episode!
        robot.start(randomize_food=True, hide_render=True)

        # Observe current state
        state = identify_food(robot.take_picture(), FOOD_HSV_MIN, FOOD_HSV_MAX)

        for _ in range(max_steps):
            # Select action (decrease noise over time)
            eps = 0.9 - (0.8 * ep / episodes)
            if np.random.random() < eps:
                action = np.random.uniform(-1, 1, 2)
            else:
                action = controller.select_action(state)

            # Perform action
            robot.move(*to_robobo_commands(action))

            # observe new state of the world
            new_state = identify_food(robot.take_picture(), FOOD_HSV_MIN, FOOD_HSV_MAX)

            # Did we find food or collide with a wall?
            collision, found_food = robot.has_collided()
            food_collected += int(found_food)

            # Compute reward
            reward = compute_reward(collision, found_food)
            rewards.append(reward)

            # learn from reward
            controller.save_experience(state, action, reward, new_state)
            controller.update()

            # End episode on collision with walls or when all food is collected
            if (collision and not found_food) or food_collected == 8:
                break

            state = new_state
            pbar.update(1)

        robot.stop()
        pbar.set_postfix({'avg_reward': np.mean(rewards), 'food_collected': food_collected})
        pbar.close()

        # Save stats accumulated over episode
        training_rewards.append(rewards)
        training_collected_foods.append(food_collected)

    # Save training stats
    with open('training_rewards.pkl', 'wb') as file:
        pickle.dump(training_rewards, file)

    with open('training_collected_foods.pkl', 'wb') as file:
        pickle.dump(training_collected_foods, file)


if __name__ == "__main__":
    # Init controller
    ddpg_controller = DDPGAgent(layer_shapes=(8, 24, 2), gamma=0.99, actor_lrate=1e-3,
                                critic_lrate=5e-3, replay_size=96)

    # Callback function to save controller on exit
    def save_controller(signal_number=None, frame=None):
        print("\nSaving controller!")
        with open('models/Task2_DDPG.pkl', 'wb') as file:
            pickle.dump(ddpg_controller, file)
        sys.exit(1)
    signal.signal(signal.SIGINT, save_controller)

    # optimize controller with DDPG
    robobo = SimulatedRobobo(ip='192.168.1.113', robot_id='')
    train_controller(robobo, ddpg_controller, max_steps=500, episodes=300)
    save_controller()
