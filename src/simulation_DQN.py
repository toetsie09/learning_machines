import math
import random
from turtle import back
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

from torch import Tensor

from DQN import DQN, ReplayMemory, Transition, Robobo_Controller


# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE = torch.device('cpu')
# print('current device:', device)
BATCH_SIZE = 5
N_ACTIONS = 3
N_INPUTS = 5
GAMMA = 0.9

def initialize_NN(input_size, num_actions, DEVICE):
    policy = DQN(input_size, num_actions, DEVICE).to(DEVICE)
    target = DQN(input_size, num_actions, DEVICE).to(DEVICE)
    target.load_state_dict(policy.state_dict())
    target.eval()
    return policy, target

def save_model(PATH, model):
    torch.save(model.state_dict(), PATH)

def load_model(PATH, input_size, num_actions, DEVICE):
    model = DQN(input_size, num_actions, DEVICE).to(DEVICE)
    model.load_state_dict(torch.load(PATH))
    model.eval()
    return model

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

    reset_data = []

    for i_episode in range(num_episodes):      
        reset_count = 0
        stuck_count = 0 
        memory = ReplayMemory()
        current_state = controller.get_state(True)

        for t_steps in range(num_steps):
            print()
            # Reverse until the front sensors are clear when the robot gets stuck
            if stuck_count > 5:
                print('Robot is stuck, fixing this now\n')
                while not front_bool:
                    controller.rob.move(-5, -5, 2000)
                    state = controller.get_state(False)
                    _, front_bool, _ = controller.detect_collision(state)
                reset_count += 1

            start_position = controller.get_position()

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

            # Store the transition in memory
            next_state = torch.Tensor(np.asarray(next_state))
            memory.push(current_state, action, next_state, reward)

            # Move to the next state
            current_state = next_state

            # Perform one step of the optimization (on the policy network)
            policy_net, target_net = optimize_model(
                policy_network=policy_net,
                target_network=target_net,
                memory_structure=memory,
                optimizer=optimizer
            )

            if (collision == 3 and back_bool) or distance < 1e-05:
                stuck_count += 1
            else:
                stuck_count = 0

        # Update the target network, copying all weights and biases in DQN
        target_net.load_state_dict(policy_net.state_dict())
        reset_data.append(reset_count)

    fig = plt.figure()
    plt.plot(reset_data, '.')
    fig.savefig('./src/figures/resets.png', dpi=fig.dpi)

    controller.terminate_simulation()

    PATH = './src/models/'
    save_model(PATH + 'DQN_policy_v1.pt', policy_net)
    save_model(PATH + 'DQN_target_v1.pt', target_net)
    
if __name__ == "__main__":
    # environment = environment_setup()
    n_episodes = 150
    n_steps = 50
    main(n_episodes, n_steps)