#!/usr/bin/env python2
from __future__ import print_function
from tqdm import tqdm

import robobo
import sys
import signal
import pickle
import random
import numpy as np

from DDPG import DDPGAgent


def get_sensor_state(env, d_max=0.25):
    # Converts this distance-based measurement from V-REP
    # into proximity measurement similar to the robot.
    values = []
    for d in env.read_irs():
        if type(d) == bool:
            values += [0]
        else:
            values += [max(0, (d_max - d) / d_max)]
    return np.array(values)


def steer_robot(env, action, millis=1000):
    angular_drive = action[0]
    l_drive = 10 * (1 + angular_drive)
    r_drive = 10 * (1 - angular_drive)
    env.move(l_drive, r_drive, millis)


def compute_reward(action, sensors, collision_thres=0.02):
    angular_drive = action[0]
    collisions = [v for v in sensors if type(v) != bool and v < collision_thres]
    if collisions:
        return -1000, True
    else:
        return 1 - abs(angular_drive), False


def train_controller(controller, max_steps_per_episode=200, episodes=20):
    # init simulated environment
    env = robobo.SimulationRobobo('#0').connect(address='192.168.1.113', port=19997)

    for ep in range(episodes):
        pbar = tqdm(total=max_steps_per_episode, position=0, leave=True, desc='Episode %s' % ep)

        # start simulation
        env.play_simulation()
        state = get_sensor_state(env)
        rewards = []

        # set random orientation of robot
        steer_robot(env, [random.choice([-1, 1])], millis=np.random.randint(0, 3000))
        env.sleep(1)

        for step in range(max_steps_per_episode):

            # epsilon-greedy policy
            p_exploit = ep / episodes
            if np.random.random() < p_exploit:
                action = controller.select_action(state)
            else:
                action = np.random.uniform(-1, 1, (1,))

            # Perform action (rotate robot)
            steer_robot(env, action)

            # observe new state and reward
            new_state = get_sensor_state(env)
            reward, done = compute_reward(action, env.read_irs())

            # learning step
            controller.save_experience(state, action, reward, new_state)
            controller.update()

            # update state
            state = new_state
            rewards.append(reward)

            if done:
                break

            pbar.set_postfix({'reward': reward})
            pbar.update(1)

        pbar.set_postfix({'avg_reward': np.mean(rewards)})

        # reset environment
        env.stop_world()
        env.wait_for_stop()


if __name__ == "__main__":
    # untrained controller
    controller = DDPGAgent(num_inputs=8, num_hidden=(32,), num_actions=1,  # angular drive
                           gamma=0.5, actor_lrate=1e-3, critic_lrate=1e-2)

    # define function to save final controller on exit
    def save_controller(signal_number=None, frame=None):
        print("\nSaving controller!")
        with open('DDPG_controller.pkl', 'wb') as file:
            pickle.dump(controller, file)
        sys.exit(1)

    signal.signal(signal.SIGINT, save_controller)

    # optimize controller with DDPG
    train_controller(controller, max_steps_per_episode=400, episodes=200)
    save_controller()

