#!/usr/bin/env python2
from __future__ import print_function

import sys
import signal
import pickle
import numpy as np

from tqdm import tqdm
from pprint import pprint
from robobo_interface import SimulatedRobobo


def terminate_program(signal_number, frame):
    print("Ctrl-C received, terminating program")
    sys.exit(1)


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


def to_robobo_commands(action, forward_drive=10, angular_drive=8):
    """ Take an action and converts it into left/right wheel
        commands for the Robobo robot.
    """
    y0, y1 = action
    left_drive = y0 * forward_drive + (1 - abs(y0)) * y1 * angular_drive
    right_drive = y0 * forward_drive - (1 - abs(y0)) * y1 * angular_drive
    return np.array([left_drive, right_drive])


def eval_controller(robot, controller, max_steps=500, episodes=10):
    # metrics
    collisions = []
    episode_lengths = []
    distance_travelled = []
    motion_smoothness = []

    for ep in range(episodes):
        robot.start(randomize_arena=True, hide_render=True)

        collided = 0
        episode_length = episodes
        positions = []
        commands = []

        for step in tqdm(range(max_steps)):

            # Observe sensor state and post-process
            state = robot.get_sensor_state()
            state = ir_to_proximity(state)

            # Select action greedily (add simulated noise)
            action = controller.select_action(state)
            action = action + np.random.normal(0, 0.1, action.shape)

            # Perform action
            command = to_robobo_commands(action)
            commands.append(command)
            robot.move(*command)

            # Record location in arena
            positions.append(robot.position)

            # End simulated episode on collision
            if robot.has_collided(d_min=0.01):
                collided = 1
                episode_length = step
                break

        robot.stop()

        # Update metrics
        collisions.append(collided)              # Did Robobo collide?
        episode_lengths.append(episode_length)   # How long was Robobo going for?

        path_length = np.sum(np.linalg.norm(np.diff(positions, axis=0), axis=1))
        distance_travelled.append(path_length)   # How far did Robobo travel

        command_mad = np.mean(np.absolute(np.diff(commands, axis=0)))
        avg_smoothness = np.exp(-command_mad)
        motion_smoothness.append(avg_smoothness) # How smooth were its movements?

    # Compute metrics from stats
    metrics = {'collision_rate': np.mean(collisions),
               'mean_time_to_collision': np.mean(episode_lengths),
               'mean_action_smoothness': np.mean(motion_smoothness),
               'mean_distance_travelled': np.mean(distance_travelled),
               'std_time_to_collision': np.std(episode_lengths),
               'std_action_smoothness': np.std(motion_smoothness),
               'std_distance_travelled': np.std(distance_travelled)}
    return metrics


if __name__ == "__main__":
    signal.signal(signal.SIGINT, terminate_program)

    # load controller from checkpoint
    with open('models/Task1_DDPG_s=5_h=24_final.pkl', 'rb') as file:
        ddpg_controller = pickle.load(file)

    # Run controller and print results
    robobo = SimulatedRobobo(ip='192.168.43.248')
    scores = eval_controller(robobo, ddpg_controller, max_steps=100, episodes=100)
    pprint(scores)
