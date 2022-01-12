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
        self.rob = robobo.SimulationRobobo().connect(address='192.168.192.14', port=19997)
        self.rob.play_simulation()
    
    def terminate_program(self, signal_number, frame):
        print("Ctrl-C received, terminating program")
        sys.exit(1)

    def select_action(self, target_network, device, n_actions, state, steps_done):
        EPS_START = 0.9
        EPS_END = 0.05
        EPS_DECAY = 200
        eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)

        e_greedy = np.random.choice(['explore', 'exploit'], p=[eps_threshold, 1-eps_threshold])
        if e_greedy == 'explore':
            return torch.tensor([[random.randrange(n_actions)]], device=device)
        else:
            return target_network(state).argmax().view(1,1) # return action with highest Q-value

    def get_reward(self, collision):
        return torch.Tensor([-100]) if collision else torch.Tensor([1])

    def get_state(self):
        return torch.Tensor(np.log(np.array(self.rob.read_irs())[-5:]) / 10)
    
    def take_action(self, action: int):
        if action == 0:
            self.move_left()
        elif action == 1:
            self.move_straight()
        else:
            self.move_right()

    def move_left(self):
        self.rob.move(5, 10, 1000)
        
    def move_right(self):
        self.rob.move(10, 5, 1000)
    
    def move_straight(self):
        self.rob.move(5, 5, 500)
        
    def detect_collision(self, state):
        values = state[~-np.isinf(state)]
        return True if len(values) > 0 and min(values) < -0.2 else False

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