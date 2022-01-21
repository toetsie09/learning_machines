#!/usr/bin/env python2
from __future__ import print_function
import sys
import signal
import pickle
import numpy as np
from tqdm import tqdm

from robobo_interface import SimulatedRobobo
from DDPG import DDPGAgent


def compute_reward(action, collision):
    """ Computes the immediate reward signal given the action
        taken by Robobo and its collisions.

        +1   for forward motion
        -1   for turning
        -100 for collision
    """
    if collision:
        return -100

    forward, turning = action
    return forward - abs(turning)


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


def to_robobo_commands(action, forward_drive=10, angular_drive=10):
    """ Take an action and converts it into left/right wheel
        commands for the Robobo robot.
    """
    y0, y1 = action
    left_drive  = y0 * forward_drive + (1 - abs(y0)) * y1 * angular_drive
    right_drive = y0 * forward_drive - (1 - abs(y0)) * y1 * angular_drive
    return left_drive, right_drive


def train_controller(robot, controller, max_steps, episodes):
    """ Train the Robobo controller in simulation with DDPG.
    """
    training_rewards = []

    for ep in range(episodes):
        robot.start(randomize_arena=True, hide_render=True)

        rewards = []

        for _ in tqdm(range(max_steps)):
            # Observe current state
            state = robot.get_sensor_state()
            state = ir_to_proximity(state)
            # state += robot.camera_features()

            # Perform action selected with epsilon-greedy (0.9 - 0.1)
            eps = 0.9 - (0.8 * ep / episodes)
            if np.random.random() < eps:
                action = np.random.uniform(-1, 1, (2,))   # random
            else:
                action = controller.select_action(state)  # policy

            robot.move(*to_robobo_commands(action))

            # observe new state of the world
            new_state = robot.get_sensor_state()
            collision = robot.has_collided()

            # Compute reward
            reward = compute_reward(action, collision)
            rewards.append(reward)

            # learn from reward
            controller.save_experience(state, action, reward, new_state)
            controller.update()

            if collision:
                break

        robot.stop()

        # Save rewards accumulated over episode
        training_rewards.append(rewards)

    return training_rewards


if __name__ == "__main__":
    # Init controller
    ddpg_controller = DDPGAgent(layer_shapes=(8, 24, 2), gamma=0.9, actor_lrate=1e-3, critic_lrate=5e-3)

    # Callback function to save controller on exit
    def save_controller(signal_number=None, frame=None):
        print("\nSaving controller!")
        with open('models/Task1_DDPG.pkl', 'wb') as file:
            pickle.dump(ddpg_controller, file)
        sys.exit(1)
    signal.signal(signal.SIGINT, save_controller)

    # optimize controller with DDPG
    robobo = SimulatedRobobo(ip='192.168.0.108', robot_id='#0')
    train_controller(robobo, ddpg_controller, max_steps=500, episodes=200)
    save_controller()
