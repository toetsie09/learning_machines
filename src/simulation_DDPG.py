#!/usr/bin/env python2
from __future__ import print_function
from tqdm import tqdm

import robobo
import sys
import signal
import numpy as np

from DDPG import DDPGAgent


def get_ir_sensors(env):
    # Convert IR distances to proximity (set 'far away' beyond d=0.2 to zero)
    values = env.read_irs()
    state = [max(0, (0.2 - d) / 0.2) if type(d) != bool else 0.0 for d in values]
    return np.array(state)


def reward_function(actions, ir_sensors, collision_thres=0.5):
    collisions = [d for d in ir_sensors if d > collision_thres]
    if collisions:
        return -np.sum(ir_sensors)
    else:
        return np.sum(actions)


def optimize_controller(controller, steps_per_episode=500, episodes=10):
    # Init simulated environment
    env = robobo.SimulationRobobo('#0').connect(address='192.168.1.113', port=19997)

    for ep in range(episodes):
        pbar = tqdm(total=steps_per_episode, position=0,
                    leave=True, desc='Episode %s' % ep)

        # Restart simulation
        env.play_simulation()
        state = get_ir_sensors(env)
        rewards = []

        # Compute exploitation probability
        p_expl = 0.1 + 0.9 * (ep / episodes)

        # Run episode
        for step in range(steps_per_episode):

            # Select action (policy or e-greedy)
            if np.random.random() < p_expl:
                action = controller.select_action(state) * 8
            else:
                action = np.random.uniform(-8, 8, 2)

            # Drive robot
            env.move(*action, 1000)

            # Observe new state and reward
            new_state = get_ir_sensors(env)
            reward = reward_function(action, new_state)

            # Update controller with experience replay
            controller.save_experience(state, action, reward, new_state)
            controller.update()

            state = new_state
            rewards.append(reward)
            pbar.set_postfix({'mean_reward': np.mean(rewards[-50:])})
            pbar.update(1)

        pbar.set_postfix({'total_reward': np.mean(rewards)})

        # Stop and reset environment
        env.stop_world()
        env.wait_for_stop()


if __name__ == "__main__":
    # Boilerplate function to terminate program properly
    def terminate_program(signal_number, frame):
        print("Ctrl-C received, terminating program")
        sys.exit(1)

    signal.signal(signal.SIGINT, terminate_program)

    # Set up template controller
    controller = DDPGAgent(num_inputs=8, num_hidden=(64, 64), num_actions=2, gamma=0.2)

    # Run optimization of controller with DDPG
    optimize_controller(controller, steps_per_episode=100, episodes=10)
