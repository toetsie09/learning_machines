#!/usr/bin/env python2
from __future__ import print_function

import sys
import signal
import pickle
import numpy as np

from tqdm import tqdm
from robobo_interface import HardwareRobobo
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


def to_robobo_commands(action, forward_drive=10, angular_drive=8):
    """ Take an action and converts it into left/right wheel
        commands for the Robobo robot.
    """
    y0, y1 = action
    left_drive = y0 * forward_drive + (1 - abs(y0)) * y1 * angular_drive
    right_drive = y0 * forward_drive - (1 - abs(y0)) * y1 * angular_drive
    return np.array([left_drive, right_drive])


def run_controller(robot, controller, calibration, max_steps=500, episodes=10):
    """ Runs the controller in hardware Robobo
    """
    for ep in range(episodes):
        for _ in tqdm(range(max_steps)):

            robot.sleep(0.6)  # Correct lag

            # Observe sensor state and post-process
            state = robot.get_sensor_state()
            state = calibration.correct_sensors(state)  # Map sensors to simulation range
            state = ir_to_proximity(state)              # Convert back from simulated range to normalized proximity
            state = state * 1.4                         # Calibrate for brown boxes

            # state += robot.camera_features()

            # Select action greedily
            action = controller.select_action(state)

            # Perform action
            command = to_robobo_commands(action)
            command[0] = command[0] * 1.35                         # Correct left motor deficit
            command = [int(round(x)) for x in command]             # Hardware requires integer input
            duration = int(900 * calibration.duration_multiplier)  # Calibrate speed relative to simulation
            robot.move(*command, millis=duration)


if __name__ == "__main__":
    signal.signal(signal.SIGINT, terminate_program)

    # load controller from checkpoint
    with open('models/Task1_DDPG_s=5_h=24_final.pkl', 'rb') as file:
        ddpg_controller = pickle.load(file)

    # load calibration model
    calibrator = Calibrator('calibration/calib_params.out')

    # Run controller and print results
    robobo = HardwareRobobo(ip='192.168.43.248')
    run_controller(robobo, ddpg_controller, calibrator, max_steps=100, episodes=100)
