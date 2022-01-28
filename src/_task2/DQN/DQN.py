### TASK 2

import random 

from collections import deque, namedtuple

import torch
from torch import relu, nn
import torch.optim as optim

class Network(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Network, self).__init__()
        self.lin1 = nn.Linear(input_size, hidden_size)
        self.lin2 = nn.Linear(hidden_size, output_size)

    def forward(self, x): 
        x = relu(self.lin1(x.float()))
        return self.lin2(x)

class DQNAgent():
    def __init__(self, n_inputs=5, n_hidden=20, n_outputs=4, gamma=0.9) -> None:
        self._n_inputs = n_inputs
        self._n_hidden = n_hidden
        self._n_outputs = n_outputs
        self._gamma = gamma

        # Initialize Networks
        self._policy_network = Network(self._n_inputs, self._n_hidden, self._n_outputs)
        self._target_network = Network(self._n_inputs, self._n_hidden, self._n_outputs)
        self._target_network.load_state_dict(self._policy_network.state_dict())
        self._target_network.eval()

        # Setup Memory
        self._memory = ReplayMemory(4096)

        # Losses and optimizers
        self._criterion = nn.SmoothL1Loss()
        self._optimizer = optim.RMSprop(self._policy_network.parameters())

    def optimize_model(self, batch_size=96):
        if len(self._memory) < batch_size:
            return 

        transitions = self._memory.sample(batch_size)

        batch = Transition(*zip(*transitions))

        # Concatenate the current states, actions and rewards
        state_batch = torch.cat(batch.state).reshape(batch_size, self._n_inputs)
        next_states = torch.cat(batch.next_state).reshape(batch_size, self._n_inputs)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # The model computes Q(s_t), then we select the columns of actions taken. 
        # These are the actions which would've been taken for each batch state according to the policy network
        state_action_values = self._policy_network(state_batch).gather(1, action_batch)
    
        # Compute expected values of actions based on the target_net; selecting their best reward
        next_state_values = self._target_network(next_states).max(1)[0].detach()

        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self._gamma) + reward_batch

        # Compute Huber loss
        loss = self._criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self._optimizer.zero_grad()
        loss.backward()
        for param in self._policy_network.parameters():
            param.grad.data.clamp_(-1, 1)
        self._optimizer.step()

    def select_action(self, current_episode, total_episodes, state):
        eps = 1 - (0.8 * current_episode / total_episodes)
        if random.random() <= eps:  # Random action (explore)
            return torch.tensor([[random.randrange(self._n_outputs)]])
        else:                       # Best action (exploit)
            with torch.no_grad():
                return self._policy_network(torch.tensor(state)).argmax().view(1,1)

    def select_best_action(self, state):
        return self._policy_network.forward(state).argmax().item()

    def save_models(self, path, name=''):
        torch.save(self._policy_network.state_dict(), f'{path}DQN_policy_network_{name}.pt')
        torch.save(self._target_network.state_dict(), f'{path}DQN_target_network_{name}.pt')
    
    def load_models(self, path):
        self._policy_network.load_state_dict(torch.load(path))
        self._policy_network.eval()

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