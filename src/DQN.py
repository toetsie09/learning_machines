import random
import math

import robobo
import signal
import sys
import torch

import numpy as np
from operator import itemgetter 

from collections import deque, namedtuple
from torch import relu, nn

from robobo.simulation import SimulationRobobo

class Robobo_Controller():
    def __init__(self) -> SimulationRobobo:
        signal.signal(signal.SIGINT, self.terminate_program)
        self.rob = robobo.SimulationRobobo('#0').connect(address='192.168.192.14', port=19997)
        self.rob.play_simulation()
    
    def terminate_program(self, signal_number, frame):
        print("Ctrl-C received, terminating program")
        sys.exit(1)

    def select_action(self, target_network, device, n_actions, state, steps_done):
        EPS_START = 1.0
        EPS_END = 0.1
        EPS_DECAY = 5000
        eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)

        e_greedy = np.random.choice(['explore', 'exploit'], p=[eps_threshold, 1-eps_threshold])
        print('action type, explore chance:', e_greedy, eps_threshold)
        if e_greedy == 'explore':
            return torch.tensor([[random.randrange(n_actions)]], device=device)
        else:
            return target_network(state).argmax().view(1,1) # return action with highest Q-value

    def get_reward(self, collision, distance, action, front_bool, back_bool):
        reward = {
            0: 0,
            1: 25,
            2: 150,
            3: 500
        }
        
        if front_bool:
            action_reward = 0.2 if action.item() == 4 else 1.2 # punish moving backwards when the front is clear
        else:
            action_reward = 1.2 if action.item() == 4 else 0.8 # reward moving backwards when the front is obstructed
        
        return torch.Tensor([distance * 3000 * action_reward - reward[collision]])

    def get_state(self, as_tensor=False):
        values = self.rob.read_irs() #[backR, backL, frontRR, frontC, frontLL]
        filter = [0, 2, 3, 5, 7]
        filtered_values = itemgetter(*filter)(values)
        return torch.Tensor(np.array(filtered_values)) if as_tensor else filtered_values

    def take_action(self, action: int):
        if action == 0: 
            self.rob.move(5, 10, 1000) # Move left
        elif action == 1: 
            self.rob.move(5, 5, 1000) # Move Forward
        elif action == 2: 
            self.rob.move(10, 5, 1000) # Move Right
        # else: 
        #     self.rob.move(-8, -8, 2000) # Move Reverse
        
    def detect_collision(self, state):
        """
            This function returns multiple types of collisions:
            Input: 
                State: List with sensor values [backR, backL, frontRR, frontC, frontLL]
            Return: 
                collision_indicator: 0 (no objects in range), 1 (object detected), 2 (object too close)
                front_bool: boolean that detects if front of robot is clear
                back_bool: boolean that detects if back of robot is clear
        """
        values = [x if x != False else np.inf for x in state]

        # Process Booleans
        back_bool = False
        front_bool = False
        if min(values[0:2]) >= 0.1:
            back_bool = True
        if min(values[-2:]) >= 0.2:
            front_bool = True

        # Process collision indicator
        object_distance = min(values)
        if object_distance == np.inf: 
            collision_indicator = 0 # Sensors read nothing, no objects nearby
        elif object_distance > 0.1: 
            collision_indicator = 1 # Objects dected, but not very close yet
        elif object_distance < 0.01: 
            collision_indicator = 3 # Object and Robot collide
        else:
            collision_indicator = 2 # Object detected, and getting very close

        return collision_indicator, front_bool, back_bool

    def get_position(self):
        return np.asarray(self.rob.position())
    
    def distance_traveled(self, pos1, pos2):
        return np.linalg.norm(pos1 - pos2) # Eucledian distance

    def terminate_simulation(self):
        # DOESNT WORK!
        self.rob.pause_simulation()
        self.rob.stop_world()
    
    def reset_simulation(self):
        # DOESNT WORK!
        self.rob.pause_simulation()
        self.rob.stop_world()
        self.rob.play_simulation()


class DQN(nn.Module):
    # Neural Network which is going to be used as function approximator as Q*
        def __init__(self, input_size, output_size, device):
            super(DQN, self).__init__()
            self.device = device
            self.lin1 = nn.Linear(input_size, 20)
            self.lin2 = nn.Linear(20, output_size)

        def forward(self, data):
            data = data.to(self.device)
            data = self.lin1(data)
            data = relu(data)
            return self.lin2(data)

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):
    """
    Container of steps performed, as a collection of 4-tuple <s, a, s_{t+1}, r>.
    This will then be used to retrieve batches of past transitions
    """
    def __init__(self, max_length=10000):
        self.memory = deque([], maxlen=max_length)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)