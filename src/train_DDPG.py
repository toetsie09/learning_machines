#!/usr/bin/env python2
from __future__ import print_function
from tqdm import tqdm

import robobo
import sys
import signal
import pickle
import numpy as np
from DDPG import DDPGAgent


def get_sensor_state(env, d_max=0.2):
    """ Converts distance-based sensor measurement from V-REP
        into (0=far, 1=close) proximity measurement.
        d_max represents the maximum distance considered (=0).
    """
    # Discard values of sensors pointing to the floor
    used = [0, 2, 3, 5, 7]
    sensor_values = [d for i, d in enumerate(env.read_irs()) if i in used]

    # Convert distances to proximity
    values = []
    for d in sensor_values:
        if type(d) == bool:
            values += [0]
        else:
            values += [max(0, (d_max - d) / d_max)]
    return np.array(values)


def has_collided(env, d_min=0.04):
    """ Checks whether a collision has occurred and where. """
    for i, d in enumerate(env.read_irs()):
        if type(d) != bool and d < d_min:
            if i < 3:
                return True, 'rear'
            else:
                return True, 'front'
    return False, None


def compute_reward(action, collision, collision_loc):
    """ Computes the immediate reward signal given the action
        taken by Robobo and its collisions.

        +1   for forward motion
        -1   for turning
        -100 for frontal collision
        -10  for backward collision
    """
    if collision and collision_loc == 'front':
        return -100
    elif collision and collision_loc == 'rear':
        return -10

    forward, turning = action
    return forward - abs(turning)


def reset_robot(env, speed=7):
    """ Randomly rotates the Robobo before starting an episode """
    rotation_time = np.random.randint(0, 8000)
    direction = np.random.choice([-speed, speed])
    env.move(direction, -direction, rotation_time)


def control_robot(env, action, millis=1000, forward_drive=10, angular_drive=10):
    """ Take an action and converts it into left/right wheel
        commands for the Robobo robot.
    """
    y0, y1 = action
    z = (y0 + 1) / 2
    l_drive = z * forward_drive + (1 - z) * y1 * angular_drive
    r_drive = z * forward_drive - (1 - z) * y1 * angular_drive
    env.move(l_drive, r_drive, millis)


def train_controller(controller, max_steps, episodes):
    """ Train the Robobo controller in simulation with DDPG.
    """
    # start simulated environment
    env = robobo.SimulationRobobo('#0').connect(address='192.168.1.113', port=19997)

    for ep in range(episodes):
        env.randomize_arena()
        env.play_simulation()
        env.toggle_visualization()
        reset_robot(env)

        # Progress bar
        pbar = tqdm(total=max_steps, position=0, desc=str(ep), leave=True)

        rewards = []
        for step in range(max_steps):
            # Observe current state
            state = get_sensor_state(env)

            # select action with decayed-epsilon-greedy
            eps = 1 - (0.8 * ep / episodes)
            if np.random.random() < eps:
                action = np.random.uniform(-1, 1, (2,))   # random
            else:
                action = controller.select_action(state)  # policy

            # perform action
            control_robot(env, action)

            # observe new state and reward
            new_state = get_sensor_state(env)
            collision, collision_loc = has_collided(env)
            reward = compute_reward(action, collision, collision_loc)
            rewards.append(reward)

            # update
            controller.save_experience(state, action, reward, new_state)
            controller.update()

            if collision:
                break

            pbar.set_postfix({'reward': reward})
            pbar.update(1)

        # Signal end of episode
        controller.save_episode_stats()
        pbar.set_postfix({'avg_reward': np.mean(rewards)})
        pbar.close()

        # reset environment
        env.stop_world()
        env.wait_for_stop()


if __name__ == "__main__":
    # untrained controller
    robot_controller = DDPGAgent(num_inputs=5, num_hidden=(24, 8), num_actions=2,  # forward motion + turning direction
                                 gamma=0.5, actor_lrate=1e-3, critic_lrate=5e-3)

    # define function to save final controller on exit
    def save_controller(signal_number=None, frame=None):
        print("\nSaving controller!")
        with open('models/Task1_DDPG.pkl', 'wb') as file:
            pickle.dump(robot_controller, file)
        sys.exit(1)

    signal.signal(signal.SIGINT, save_controller)

    # optimize controller with DDPG
    train_controller(robot_controller, max_steps=500, episodes=400)
    save_controller()
