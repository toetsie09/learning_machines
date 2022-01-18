#!/usr/bin/env python2
from __future__ import print_function

import robobo
import sys
import signal
import pickle
import numpy as np

from train_DDPG import reset_robot, control_robot, get_sensor_state, has_collided
from pprint import pprint


def eval_controller(controller, env_type='simulation', randomize_arena=False, max_steps=500, episodes=10):
    # Init simulated or hardware environment
    if env_type == 'simulation':
        env = robobo.SimulationRobobo('#0').connect(address='192.168.1.113', port=19997)
    elif env_type == 'hardware':
        env = robobo.HardwareRobobo(camera=False).connect(address="<ADDRESS HERE>")  # TODO: update address
    else:
        raise Exception('env_type %s not supported' % env_type)

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
        actions_taken = []
        positions = []

        # (Optionally) Start V-REP
        if env_type == 'simulation':
            if randomize_arena:
                env.randomize_arena()
            env.play_simulation()
            reset_robot(env)

        # Control robot with policy
        for step in range(max_steps):
            state = get_sensor_state(env)

            action = controller.select_action(state)
            actions_taken.append(action)

            action += np.random.normal(0, 0.08, action.shape)  # to fix stuckiness
            control_robot(env, action)

            if env_type == 'simulation':
                positions.append(env.position())

            if has_collided(env, d_min=0.01):
                collision = 1
                collision_time = step
                break

        # (Optionally) Stop V-REP
        if env_type == 'simulation':
            env.stop_world()
            env.wait_for_stop()

        # Did I collide?
        collisions.append(collision)

        # Time-to-collision
        time_to_collision.append(collision_time)

        # How smooth was my movement
        action_mad = np.mean(np.absolute(np.diff(actions_taken, axis=0)))
        motion_smoothness.append(np.exp(-action_mad))

        # How far did I travel
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
    with open('models/Task1_DDPG_s=5_h=24-8.pkl', 'rb') as file:
        agent = pickle.load(file)

    # optimize controller with DDPG
    scores = eval_controller(agent, env_type='simulation_random', randomize_arena=True,
                             max_steps=1000, episodes=1)
    pprint(scores)
