#!/usr/bin/env python2
from __future__ import print_function

import robobo
import sys
import signal
import pickle
import numpy as np

from train_DDPG import RoboboEnv, to_robobo_commands
from pprint import pprint


def eval_controller(env, controller, max_steps=500, episodes=10):
    # Define statistics
    collisions = []
    time_to_collision = []
    distance_travelled = []
    motion_smoothness = []

    # Start environment
    for ep in range(episodes):
        # Record stats during episode
        collision = 0
        collision_time = episodes
        positions, actions = [], []

        env.start()
        for step in range(max_steps):
            # Observe state
            state = env.get_sensor_state()
            state[:2] = state[:2] * 0.0  # Bugfix: distance estimation of rear sensors

            if env.is_simulation:
                positions.append(env.robot_position())

            # Select action greedily (add noise to fix stuck robot)
            action = controller.select_action(state)
            action2 = action + np.random.normal(0, 0.1, action.shape)
            actions.append(action)

            # Perform action
            env.move_robot(*to_robobo_commands(action2))

            # End if collision
            if env.has_collided(d_min=0.01)[0]:
                collision = 1
                collision_time = step
                break
        env.stop()

        # Count scratches
        collisions.append(collision)

        # How long was it going for?
        time_to_collision.append(collision_time)

        # How smooth were its movements?
        action_mad = np.mean(np.absolute(np.diff(actions, axis=0)))
        avg_smoothness = np.exp(-action_mad)
        motion_smoothness.append(avg_smoothness)

        # How far did it travel over the course of the episode?
        path_length = np.sum(np.linalg.norm(np.diff(positions, axis=0), axis=1))
        distance_travelled.append(path_length)

    metrics = {'collision_rate': np.mean(collisions),
               'avg_time_to_collision': np.mean(time_to_collision),
               'action_smoothness': np.mean(motion_smoothness),
               'distance_travelled': np.mean(distance_travelled)}
    return metrics


if __name__ == "__main__":
    # Boilerplate function to terminate program properly
    def terminate_program(signal_number, frame):
        print("Ctrl-C received, terminating program")
        sys.exit(1)

    signal.signal(signal.SIGINT, terminate_program)

    # load trained controller from pickled checkpoint
    with open('models/Task1_DDPG_s=5_h=24.pkl', 'rb') as file:
        agent = pickle.load(file)

    # optimize controller with DDPG
    robobo = RoboboEnv(env_type='randomized_simulation', robot_id='#0', ip='192.168.1.113')
    scores = eval_controller(robobo, agent, env_type='simulation', randomize_arena=True,
                             max_steps=1000, episodes=5)
    pprint(scores)
