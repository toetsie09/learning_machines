#!/usr/bin/env python2
from __future__ import print_function

import robobo
import sys
import signal
import pickle

from DDPG import DDPGAgent
from train_DDPG import *


def test_controller(controller, max_steps=500, episodes=10):
    # Init simulated environment
    env = robobo.SimulationRobobo('#0').connect(address='192.168.1.113', port=19997)

    for ep in range(episodes):
        env.play_simulation()
        reset_robot(env)

        # Run robot with controller
        for step in range(max_steps):
            state = get_sensor_state(env)
            action = controller.select_action(state)
            action += np.random.normal(0, 0.08, action.shape)
            control_robot(env, action)

        env.stop_world()
        env.wait_for_stop()


if __name__ == "__main__":
    # Boilerplate function to terminate program properly
    def terminate_program(signal_number, frame):
        print("Ctrl-C received, terminating program")
        sys.exit(1)

    signal.signal(signal.SIGINT, terminate_program)

    # load trained controller
    with open('DDPG_controller.pkl', 'rb') as file:
        controller = pickle.load(file)

    # optimize controller with DDPG
    test_controller(controller, max_steps=500, episodes=20)

