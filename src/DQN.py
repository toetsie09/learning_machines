import random
import math

import robobo
import signal
import sys
import torch

import numpy as np

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
        EPS_START = 0.9
        EPS_END = 0.05
        EPS_DECAY = 10000
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
            1: 40,
            2: 400,
            3: 500
        }
        if front_bool:
            action_reward = 0.5 if action.item() == 4 else 1.5 # punish moving backwards when the front is clear
        else:
            action_reward = 1.5 if action.item() == 4 else 0.5 # reward moving backwards when the front is obstructed
        
        return torch.Tensor([distance * 1000 * action_reward - reward[collision]])

    def get_state(self, as_tensor=False):
        if as_tensor:
            return torch.Tensor(np.array(self.rob.read_irs()))
        else:
            return self.rob.read_irs()
    
    def take_action(self, action: int):
        if action == 0:
            self.move_left()
        elif action == 1:
            self.move_straight()
        elif action == 2:
            self.move_right()
        else:
            self.move_reverse()

    def move_left(self):
        self.rob.move(5, 10, 2000)
        
    def move_right(self):
        self.rob.move(10, 5, 2000)
    
    def move_straight(self):
        self.rob.move(8, 8, 2000)
    
    def move_reverse(self):
        self.rob.move(-8, -8, 2000)
        
    def detect_collision(self, state):
        """
            This function returns multiple types of collisions:
            Return: 
                collision_indicator: 0 (no objects in range), 1 (object detected), 2 (object close), 3 (object collision)
                front_bool: boolean that detects if front of robot is clear
                back_bool: boolean that detects if back of robot is clear
        """
        back_bool = False
        front_bool = False
        # print('state', state) 
        values = [x if x != False else np.inf for x in state]
        # print('values', values)
        if min(values[0:3]) >= 0.1:
            back_bool = True
        if min(values[-5:]) >= 0.1:
            front_bool = True

        object_distance = min(values)
        if object_distance == np.inf:
            collision_indicator = 0
        elif object_distance > 0.1:
            collision_indicator = 1
        elif object_distance < 0.01:
            collision_indicator = 3
        else:
            collision_indicator = 2

        return collision_indicator, front_bool, back_bool


    def get_position(self):
        return np.asarray(self.rob.position())
    
    def distance_traveled(self, pos1, pos2):
        # Eucledian distance
        return np.linalg.norm(pos1 - pos2)

    def terminate_simulation(self):
        self.rob.pause_simulation()
        self.rob.stop_world()
    
    def reset_simulation(self):
        self.rob.pause_simulation()
        self.rob.stop_world()
        self.rob.play_simulation()


class DQN(nn.Module):
    # Neural Network which is going to be used as function approximator as Q*
        def __init__(self, input_size, output_size, device):
            super(DQN, self).__init__()
            self.device = device
            self.lin1 = nn.Linear(input_size, 50)
            self.lin2 = nn.Linear(50, output_size)

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