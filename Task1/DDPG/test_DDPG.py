#!/usr/bin/env python2
from __future__ import print_function

import sys
import signal
import pickle
import numpy as np
import time

from pprint import pprint
from robot_interface import RoboboEnv
from calibration.calibrate_robobo import Calibrator


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


def select_action(controller, state):
    action1 = controller.select_action(state)

    # Invert world
    if len(state) == 8:
        state2 = state[[2, 1, 0, 7, 6, 5, 4, 3]]
    else:
        state2 = state[[1, 0, 4, 3, 2]]
    action2 = controller.select_action(state2)

    # Select normal action if appropriate
    if np.linalg.norm(action1) > np.linalg.norm(action2):
        return action1

    # Select alternative rotation if more safe
    action2[1] = -action2[1]
    return action2


def to_robobo_commands(action, forward_drive=10, angular_drive=8):
    """ Take an action and converts it into left/right wheel
        commands for the Robobo robot.
    """
    y0, y1 = action
    z = (y0 + 1) / 2
    l_drive = z * forward_drive + (1 - z) * y1 * angular_drive
    r_drive = z * forward_drive - (1 - z) * y1 * angular_drive
    return np.array([l_drive, r_drive])


def eval_controller(robot, controller, calibration, max_steps=500, episodes=10):
    # metrics
    collisions = []
    time_to_collision = []
    distance_travelled = []
    motion_smoothness = []

    for ep in range(episodes):

        collision = 0
        collision_time = -1
        positions, actions = [], []

        robot.start()
        for step in range(max_steps):

            # Observe sensor state and post-process
            #robot._env.sleep(seconds=1)
            state = robot.get_sensor_state()
            if not robot.in_simulation:  # Running actual hardware
                state = calibration.correct_sensors(state)
            state = ir_to_proximity(state)

            # Bugfix: cap values of tilted sensors to prevent ground detection
            if not robot.in_simulation and len(state) == 8:
                state[1] = state[4] if state[4] > 0.32 else 0.0  # Back-C
                state[4] = state[4] if state[4] > 0.32 else 0.0  # Front-L
                state[6] = state[6] if state[6] > 0.32 else 0.0  # Front-R
            print('state:', state)

            # Select action greedily (add simulated sensor noise)
            action = select_action(controller, state)
            actions.append(action)
            action = action + np.random.normal(0, 0.1, action.shape)

            # Perform action
            command = to_robobo_commands(action)
            if not robot.in_simulation:
                command[0] = command[0] * 1.35  # Correct relative speed motors
                command = [int(round(x)) for x in command]  # Hardware requires integer input
                duration = int(800 * calibration.duration_multiplier)
                robot.move(*command, millis=duration)
            else:
                robot.move(*command)

            if robot.in_simulation:
                positions.append(robot.position)

            # End episode on collision
            if robot.has_collided(d_min=0.01) is not None:
                collision = 1
                collision_time = step
                break
        robot.stop()

        # Did Robobo crash?
        collisions.append(collision)

        # How long was Robobo going for?
        time_to_collision.append(collision_time)

        # How smooth were its movements?
        action_mad = np.mean(np.absolute(np.diff(actions, axis=0)))
        avg_smoothness = np.exp(-action_mad)
        motion_smoothness.append(avg_smoothness)

        # How far did it travel over the course of the episode?
        if robot.in_simulation:
            path_length = np.sum(np.linalg.norm(np.diff(positions, axis=0), axis=1))
        else:
            path_length = -1
        distance_travelled.append(path_length)

    # Compute metrics from stats
    metrics = {'collision_rate': np.mean(collisions),
               'avg_time_to_collision': np.mean(time_to_collision),
               'action_smoothness': np.mean(motion_smoothness),
               'distance_travelled': np.mean(distance_travelled)}
    return metrics


if __name__ == "__main__":
    signal.signal(signal.SIGINT, terminate_program)

    MODEL_FILE = 'models/Task1_DDPG_s=5_h=24_final.pkl'  # TODO: specify model

    # load controller from checkpoint
    with open(MODEL_FILE, 'rb') as file:
        ddpg_controller = pickle.load(file)

    # load calibration model
    calibration = Calibrator('calibration/calib_params.out')

    # Run controller and print results
    robobo = RoboboEnv(env_type='hardware', ip='192.168.0.115', used_sensors='level')
    scores = eval_controller(robobo, ddpg_controller, calibration=calibration,
                             max_steps=1000, episodes=100)
    pprint(scores)
