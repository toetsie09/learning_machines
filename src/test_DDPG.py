#!/usr/bin/env python2
from __future__ import print_function

import sys
import signal
import pickle
import numpy as np
import cv2

from tqdm import tqdm
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

    if len(state) == 8:
        state2 = state[[2, 1, 0, 7, 6, 5, 4, 3]]
    else:
        state2 = state[[1, 0, 4, 3, 2]]
    action2 = controller.select_action(state2)

    if np.linalg.norm(action1) > np.linalg.norm(action2):
        return action1

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


def eval_controller(robot:RoboboEnv, controller, calibration:Calibrator, max_steps=500, episodes=10):
    # metrics
    collisions = []
    time_to_collision = []
    distance_travelled = []
    motion_smoothness = []
    
    robot._env.set_emotion("sad")
    robot._env.set_phone_tilt(109,100)
    #virtual env
    #robot.sleep(0.5)
    robot._env.talk("I going for a ride.")
    robot._env.set_emotion("happy")
    
    for ep in range(episodes):

        collision = 0
        collision_time = episodes
        positions, commands = [], []

        robot.start()
        for step in tqdm(range(max_steps)):
            image = robot._env.get_image_front()
            image = image[160:, :] #crop lowersection HxW 640x480 -> 480x480
            image = cv2.medianBlur(image,ksize=5) # smooth image to denoise it

            if  robot.in_simulation:
                cv2.destroyAllWindows()
                cv2.imshow("Live view",image)
            else:
                cv2.imwrite(f"./src/view/test_picture_{ep}_{step}.png",image)

            # Correct lag
            robot.sleep(0.1)

            # Observe sensor state and post-process
            state = robot.get_sensor_state()
            if not robot.in_simulation:  # Running actual hardware
                state = calibration.correct_sensors(state)
                state = ir_to_proximity(state)
                state = state * 1.4  # Calibrate for brown boxes
            else:
                state = ir_to_proximity(state)

            # Select action greedily (add simulated sensor noise)
            action = select_action(controller, state)
            action = action + np.random.normal(0, 0.1, action.shape)

            # Perform action
            command = to_robobo_commands(action)
            commands.append(command)
            if not robot.in_simulation:  # Actual hardware
                command[0] = command[0] * 1.35  # Correct relative speed of motors
                command = [int(round(x)) for x in command]  # Hardware requires integer input
                duration = int(800 * calibration.duration_multiplier)  # Calibrate speed relative to simulation
                robot.move(*command, millis=duration)
            else:
                robot.move(*command)

            # Record location in arena (if simulated)
            if robot.in_simulation:
                positions.append(robot.position)

            # End simulated episode on collision
            if robot.in_simulation and robot.has_collided(d_min=0.01):
                collision = 1
                collision_time = step
                break
        robot.stop()

        if robot.in_simulation:
            # Did Robobo crash?
            collisions.append(collision)

            # How long was Robobo going for?
            time_to_collision.append(collision_time)

            # How smooth were its movements?
            path_length = np.sum(np.linalg.norm(np.diff(positions, axis=0), axis=1))
            distance_travelled.append(path_length)

            # How smooth were its movements?
            command_mad = np.mean(np.absolute(np.diff(commands, axis=0)))
            avg_smoothness = np.exp(-command_mad)
            motion_smoothness.append(avg_smoothness)

    # Compute metrics from stats
    if robot.in_simulation:
        metrics = {'collision_rate': np.mean(collisions),
                   'mean_time_to_collision': np.mean(time_to_collision),
                   'mean_action_smoothness': np.mean(motion_smoothness),
                   'mean_distance_travelled': np.mean(distance_travelled),
                   'std_time_to_collision': np.std(time_to_collision),
                   'std_action_smoothness': np.std(motion_smoothness),
                   'std_distance_travelled': np.std(distance_travelled)}
        return metrics
    return {}


if __name__ == "__main__":
    signal.signal(signal.SIGINT, terminate_program)

    MODEL_FILE = './src/models/Task1_DDPG_s=5_h=24_final.pkl'  # TODO: specify model
    IP = '192.168.2.11'                                  # TODO: specify IP

    # load controller from checkpoint
    with open(MODEL_FILE, 'rb') as file:
        ddpg_controller = pickle.load(file)

    # load calibration model
    calibration = Calibrator('./src/calibration/calib_params.out')

    # Run controller and print results
    robobo = RoboboEnv(env_type='hardware', ip=IP, used_sensors='5')
    #robobo = RoboboEnv(env_type='randomized_simulation', ip=IP, used_sensors='5', hide_render=True)
    scores = eval_controller(robobo, ddpg_controller, calibration=calibration, max_steps=333, episodes=1)
    pprint(scores)
