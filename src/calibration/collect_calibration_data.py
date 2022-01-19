#!/usr/bin/env python2
from __future__ import print_function
import sys
sys.path.insert(0, '../')

import robobo
import sys
import signal
import numpy as np


def terminate_program(signal_number, frame):
    print("Ctrl-C received, terminating program")
    sys.exit(1)


# TODO: determine at what value there's a collision in the hardware robot
def has_collided(env, env_type, simulation_thres=0.01, hardware_thres=0.99):
    if env_type == 'simulation':
        collisions = [x for x in env.read_irs() if type(x) != bool and x < simulation_thres]
    else:
        collisions = [x for x in env.read_irs() if type(x) != bool and x > hardware_thres]
    return any(collisions)


def get_calibration_data(env, env_type='simulation', samples=50):
    # Start simulation (if enabled)
    if env_type == 'simulation':
        env.play_simulation()

    # Move until collision
    sensor_readings = []
    while not has_collided(env, env_type):
        env.move(2, 2, 500)
        env.sleep(1)

        # Register several measurements at each position
        readings = [float(env.read_irs()[5]) for _ in range(samples)]
        sensor_readings.append(readings)
    print('Ouch')

    # Stop sim (if enabled)
    if env_type == 'simulation':
        env.stop_world()
        env.wait_for_stop()

    return np.array(sensor_readings)


if __name__ == "__main__":
    signal.signal(signal.SIGINT, terminate_program)

    IP = '192.168.1.113'  # TODO: update address when changing networks!
    ENV_TYPE = 'hardware'  # or 'simulation'

    if ENV_TYPE == 'simulation':
        input('V-REP open with calibration_scene.ttt loaded? (press ENTER)')
        env = robobo.SimulationRobobo().connect(address=IP, port=19997)

    elif ENV_TYPE == 'hardware':
        input('\nRobobo placed 1x its length from a wall? (press ENTER)')
        env = robobo.HardwareRobobo(camera=False).connect(address=IP)
    else:
        raise Exception('ENV_TYPE {} not supported'.format(ENV_TYPE))

    # Record distances at regular time-intervals
    dists = get_calibration_data(env, env_type=ENV_TYPE)
    np.savetxt('sensor_calib_{}.out'.format(ENV_TYPE), dists)



