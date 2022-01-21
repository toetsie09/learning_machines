#!/usr/bin/env python2
from __future__ import print_function
import sys
import signal
import pickle
import numpy as np
from tqdm import tqdm

from robot_interface import RoboboEnv
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
    z = (y0 + 1) / 2
    l_drive = z * forward_drive + (1 - z) * y1 * angular_drive
    r_drive = z * forward_drive - (1 - z) * y1 * angular_drive
    return l_drive, r_drive


def train_controller(robot, controller, max_steps, episodes):
    """ Train the Robobo controller in simulation with DDPG.
    """
    for ep in range(episodes):
        pbar = tqdm(total=max_steps, position=0, desc=str(ep), leave=True)
        rewards = []

        robot.start()
        for step in range(max_steps):
            # Observe current state
            state = robot.get_sensor_state()
            state = ir_to_proximity(state)

            # Perform action selected with epsilon-greedy
            eps = 1 - (0.8 * ep / episodes)
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

            pbar.set_postfix({'reward': reward})
            pbar.update(1)

        # End episode
        robot.stop()
        controller.save_episode_stats()
        pbar.set_postfix({'avg_reward': np.mean(rewards)})
        pbar.close()


if __name__ == "__main__":
    # untrained controller
    ddpg_controller = DDPGAgent(num_inputs=8, num_hidden=(24,), num_actions=2,  # forward motion + turning direction
                                gamma=0.5, actor_lrate=1e-3, critic_lrate=5e-3)

    # define function to save final controller on exit
    def save_controller(signal_number=None, frame=None):
        print("\nSaving controller!")
        with open('models/Task1_DDPG.pkl', 'wb') as file:
            pickle.dump(ddpg_controller, file)
        sys.exit(1)

    signal.signal(signal.SIGINT, save_controller)

    # optimize controller with DDPG
    robobo = RoboboEnv(env_type='randomized_simulation', robot_id='#0', ip='192.168.0.108')  # 192.168.0.108
    train_controller(robobo, ddpg_controller, max_steps=500, episodes=200)
    save_controller()
