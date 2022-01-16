import math
import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
import seaborn as sns

from torch import Tensor

from DQN import DQN, ReplayMemory, Transition, Robobo_Controller

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE = torch.device('cpu')
# print('current device:', device)
BATCH_SIZE = 5
N_ACTIONS = 4
N_INPUTS = 8
GAMMA = 0.9

def initialize_NN(input_size, num_actions, DEVICE):
    # Get screen size so that we can initialize layers correctly based on shape
    # returned from AI gym. Typical dimensions at this point are close to 3x40x90
    # which is the result of a clamped and down-scaled render buffer in get_screen()
    policy = DQN(input_size, num_actions, DEVICE).to(DEVICE)
    target = DQN(input_size, num_actions, DEVICE).to(DEVICE)
    target.load_state_dict(policy.state_dict())
    target.eval()
    return policy, target


def compute_loss(output, expected_output):
    """ Compute Huber loss """
    loss = nn.SmoothL1Loss()
    return loss(output, expected_output)


def expected_q_value(target_network, next_states, rewards):
    """This functions computes the q-value with the target network, which represents the Q* of the given envioronment"""
    # Compute V(s_{t+1}) for all next states. Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net. This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = target_network(next_states).max(1)[0].detach()
    # Compute the expected Q values, based on the V values from the target network
    return (next_state_values * GAMMA) + rewards


def computed_q_value(policy_network, current_states, actions):
    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the columns of actions taken.
    # These are the actions which would've been taken for each batch state according to policy_net
    return policy_network(current_states).gather(1, actions)


def optimize_model(policy_network, target_network, memory_structure, optimizer):
    if len(memory_structure) < BATCH_SIZE:
        return policy_network, target_network

    # Retrieve BATCH_SIZE number of transitions
    transitions = memory_structure.sample(BATCH_SIZE)
    # This converts batch-array of Transitions to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Concatenate the current states, actions and rewards
    state_batch = torch.cat(batch.state).reshape(BATCH_SIZE, N_INPUTS)
    next_states = torch.cat(batch.next_state).reshape(BATCH_SIZE, N_INPUTS)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    state_action_values = computed_q_value(policy_network=policy_network,
                                           current_states=state_batch,
                                           actions=action_batch)

    expected_state_action_values = expected_q_value(target_network=target_network,
                                                    next_states=next_states,
                                                    rewards=reward_batch)

    loss = compute_loss(state_action_values, expected_state_action_values.unsqueeze(1))

    # Backpropagation
    optimizer.zero_grad()
    loss.backward()
    for param in policy_network.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()
    return policy_network, target_network

def get_q_values(target_network, num_states: int):
    q_values = []
    with torch.no_grad():
        for state in range(num_states):
            q_val = target_network(torch.Tensor([state])).view(2, 1)
            q_values.append(q_val)
    q_values = torch.cat(q_values, dim=1).numpy()

def main(num_episodes: int, num_steps: int):
    ## Connect to robot
    controller = Robobo_Controller()
    steps_done = 0

    policy_net, target_net = initialize_NN(N_INPUTS, N_ACTIONS, DEVICE)
    optimizer = optim.Adam(policy_net.parameters())

    # for i in range(5):
    #     controller.move_straight()

    #     current_state = controller.get_state()
    #     print(current_state)

    # controller.move_left()

    # next_state = controller.get_state()
    # print(next_state)

    # controller.terminate()

    for i_episode in range(num_episodes):
        memory = ReplayMemory()

        current_state = controller.get_state(True)

        for t_steps in range(num_steps):
            start_position = controller.get_position()
            print(start_position)
            # Select and perform an action
            action = controller.select_action(target_net, DEVICE, N_ACTIONS, current_state, steps_done)
            print('action:', action)

            controller.take_action(action)
            end_position = controller.get_position()
            steps_done += 1
            next_state = controller.get_state(as_tensor=False)
            collision, front_bool, back_bool = controller.detect_collision(next_state)
            print('collision:', collision)
            distance = controller.distance_traveled(start_position, end_position)
            print('distance traveled', distance)

            reward = controller.get_reward(collision, distance, action, front_bool, back_bool)
            print('reward:', reward)

            # if collision:
            #     next_state = None

            # Store the transition in memory
            next_state = torch.Tensor(np.asarray(next_state))
            memory.push(current_state, action, next_state, reward)

            # Move to the next state
            current_state = next_state

            # print(policy_net, '\n')
            # print(target_net)

            # Perform one step of the optimization (on the policy network)
            policy_net, target_net = optimize_model(
                policy_network=policy_net,
                target_network=target_net,
                memory_structure=memory,
                optimizer=optimizer
            )

        # Update the target network, copying all weights and biases in DQN
        target_net.load_state_dict(policy_net.state_dict())
    controller.terminate_simulation()
    
if __name__ == "__main__":
    # environment = environment_setup()
    n_episodes = 100
    n_steps = 100
    main(n_episodes, n_steps)