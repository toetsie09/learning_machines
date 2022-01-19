#!/usr/bin/env python2
from __future__ import print_function
from tqdm import tqdm

import robobo
import sys
import signal
import pickle
import numpy as np
from DDPG import DDPGAgent


class RoboboEnv:
    def __init__(self, env_type='simulation', robot_id='#0', ip='192.168.1.113', show=False):
        # Init simulated or hardware arena
        if env_type == 'simulation' or env_type == 'randomized_simulation':
            self._env = robobo.SimulationRobobo(robot_id).connect(address=ip, port=19997)
        elif env_type == 'hardware':
            self._env = robobo.HardwareRobobo(camera=False).connect(address=ip)
        else:
            raise Exception('env_type %s not supported' % env_type)

        # Important parameters
        self._ir_sensors = [0, 2, 3, 5, 7]
        self._randomize_arena = 'randomized' in env_type
        self._is_simulation = 'simulation' in env_type
        self._show = show

    ## Utilities

    @property
    def is_simulation(self):
        return self._is_simulation

    def start(self):
        """ Start simulation in V-REP """
        if self._is_simulation:
            if self._randomize_arena:
                self._env.randomize_arena()
            self._env.play_simulation()

            if not self._show:
                self._env.toggle_visualization()
            self.reset_robot()

    def reset_robot(self, speed=7):
        """ Randomly rotates the Robobo before starting an episode """
        duration = np.random.randint(0, 8000)
        direction = np.random.choice([-speed, speed])
        self.move_robot(direction, -direction, duration)

    def robot_position(self):
        return self._env.position()

    def stop(self):
        """ Stops simulation in V-REP """
        if self._is_simulation:
            self._env.stop_world()
            self._env.wait_for_stop()

    def has_collided(self, d_min=0.04):
        """ Checks whether a collision has occurred and where (front/rear). """
        for i, d in enumerate(self._env.read_irs()):
            if type(d) != bool and d < d_min:
                if i < 3:
                    return True, 'rear'
                else:
                    return True, 'front'
        return False, None

    ## States, Actions and Rewards

    def get_sensor_state(self, d_max=0.2):
        """ Converts distance-based sensor measurement from V-REP
            into (0=far, 1=close) proximity measurement.
            d_max represents the maximum distance considered (=0).
        """
        # Discard values of IR sensors pointing to the floor
        raw_sensors = self._env.read_irs()
        sensor_values = [d for i, d in enumerate(raw_sensors) if i in self._ir_sensors]

        # Convert distances to proximity
        values = []
        for d in sensor_values:
            if type(d) == bool:
                values += [0]
            else:
                values += [max(0, (d_max - d) / d_max)]
        return np.array(values)

    def move_robot(self, left, right, millis=1000):
        self._env.move(left, right, millis)

    @staticmethod
    def compute_reward(action, collision, collision_loc):
        """ Computes the immediate reward signal given the action
            taken by Robobo and its collisions.

            +1   for forward motion
            -1   for turning
            -100 for front collision
            -10  for rear collision
        """
        if collision and collision_loc == 'front':
            return -100
        elif collision and collision_loc == 'rear':
            return -10

        forward, turning = action
        return forward - abs(turning)


def to_robobo_commands(action, forward_drive=10, angular_drive=10):
    """ Take an action and converts it into left/right wheel
        commands for the Robobo robot.
    """
    y0, y1 = action
    z = (y0 + 1) / 2
    l_drive = z * forward_drive + (1 - z) * y1 * angular_drive
    r_drive = z * forward_drive - (1 - z) * y1 * angular_drive
    return l_drive, r_drive


def train_controller(env, controller, max_steps, episodes):
    """ Train the Robobo controller in simulation with DDPG.
    """
    for ep in range(episodes):
        pbar = tqdm(total=max_steps, position=0, desc=str(ep), leave=True)
        rewards = []

        env.start()
        for step in range(max_steps):
            # Observe current state
            state = env.get_sensor_state()

            # Perform action selected with epsilon-greedy
            eps = 1 - (0.8 * ep / episodes)
            if np.random.random() < eps:
                action = np.random.uniform(-1, 1, (2,))   # random
            else:
                action = controller.select_action(state)  # policy

            env.move_robot(*to_robobo_commands(action))

            # observe new state and compute reward
            new_state = env.get_sensor_state()
            collision, collision_loc = env.has_collided()
            reward = env.compute_reward(action, collision, collision_loc)
            rewards.append(reward)

            # update
            controller.save_experience(state, action, reward, new_state)
            controller.update()

            if collision:
                break

            pbar.set_postfix({'reward': reward})
            pbar.update(1)

        # Record stats of episode
        controller.save_episode_stats()
        pbar.set_postfix({'avg_reward': np.mean(rewards)})
        pbar.close()
        env.stop()


if __name__ == "__main__":
    # untrained controller
    ddpg_controller = DDPGAgent(num_inputs=5, num_hidden=(24,), num_actions=2,  # forward motion + turning direction
                                gamma=0.5, actor_lrate=1e-3, critic_lrate=5e-3)

    # define function to save final controller on exit
    def save_controller(signal_number=None, frame=None):
        print("\nSaving controller!")
        with open('models/Task1_DDPG.pkl', 'wb') as file:
            pickle.dump(ddpg_controller, file)
        sys.exit(1)

    signal.signal(signal.SIGINT, save_controller)

    # optimize controller with DDPG
    robobo = RoboboEnv(env_type='randomized_simulation', robot_id='#0', ip='192.168.1.113')
    train_controller(robobo, ddpg_controller, max_steps=500, episodes=200)
    save_controller()
