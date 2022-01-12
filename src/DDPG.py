import numpy as np
import random

import torch
from torch import nn, optim
from torch.autograd import Variable
import torch.nn.functional as F

from collections import deque


class FCNN(nn.Module):
    def __init__(self, inputs=5, hidden=(), outputs=2, role='actor'):
        super().__init__()
        self._role = role
        self._shape = (inputs,) + hidden + (outputs,)
        self._layers = nn.ModuleList()
        for i in range(len(self._shape) - 1):
            in_ = self._shape[i]
            out = self._shape[i + 1]
            self._layers.append(nn.Linear(in_, out, bias=True))

    def copy_weights(self, other):
        for param, param_value in zip(self.parameters(), other.parameters()):
            param.data.copy_(param_value.data)
        return self

    def forward(self, x):
        # Forward pass through network
        for layer in self._layers[:-1]:
            x = F.relu(layer(x))
        out = self._layers[-1](x)

        if self._role == 'actor':
            out = torch.tanh(out)

        return out


class ReplayBuffer:
    def __init__(self, max_size=4096, batch_size=256):
        self._max_size = max_size
        self._batch_size = min(batch_size, max_size)  # batch_size <= max_size!
        self._buffer = deque(maxlen=max_size)

    def push(self, state, action, reward, next_state):
        experience = (state, action, [reward], next_state)
        self._buffer.append(experience)

    def sample(self):
        # Sample batch of experience from replay buffer
        batch_size = min(self._batch_size, len(self._buffer))
        experience = random.sample(self._buffer, k=batch_size)

        states, actions, rewards, next_states = zip(*experience)
        states = torch.FloatTensor(np.array(states))
        actions = torch.FloatTensor(np.array(actions))
        rewards = torch.FloatTensor(np.array(rewards))
        next_states = torch.FloatTensor(np.array(next_states))

        return states, actions, rewards, next_states


class DDPGAgent:
    def __init__(self, num_inputs=5, num_hidden=(), num_actions=2, actor_lrate=1e-4, critic_lrate=1e-3,
                 gamma=0.9, tau=1e-2, max_replay_buffer_size=1028, replay_size=128):
        # Hyper-parameters
        self._num_inputs = num_inputs
        self._num_actions = num_actions
        self._gamma = gamma
        self._tau = tau

        # Store experience in replay buffer
        self._replay_buffer = ReplayBuffer(max_replay_buffer_size, replay_size)

        # Actor, Critic and Target networks
        self._actor = FCNN(num_inputs, num_hidden, num_actions, role='actor')
        self._actor_target = FCNN(num_inputs, num_hidden, num_actions, role='actor')
        self._actor_target.copy_weights(self._actor)

        self._critic = FCNN(num_inputs + num_actions, num_hidden, 1, role='critic')
        self._critic_target = FCNN(num_inputs + num_actions, num_hidden, 1, role='critic')
        self._critic_target.copy_weights(self._critic)

        # Losses and optimizers
        self._critic_loss = nn.MSELoss()
        self._actor_optim = optim.Adam(self._actor.parameters(), lr=actor_lrate)
        self._critic_optim = optim.Adam(self._critic.parameters(), lr=critic_lrate)

    def save_experience(self, state, action, reward, next_state):
        self._replay_buffer.push(state, action, reward, next_state)

    def select_action(self, state):
        # Infer action using Actor
        state = Variable(torch.from_numpy(state).float().unsqueeze(0))  # Tensor([[x0, x1, x2, ...]])
        action = self._actor(state)
        return action.detach().numpy()[0]

    def update(self):
        # Get batch from replay buffer (if buffer is large enough)
        states, actions, rewards, next_states = self._replay_buffer.sample()

        # Critic loss
        critic_input = torch.cat((states, actions), dim=1)
        critic_vals = self._critic(critic_input)
        next_actions = self._actor_target(next_states)

        critic_target_input = torch.cat((next_states, next_actions.detach()), dim=1)
        next_critic_vals = self._critic_target(critic_target_input)
        yi = rewards + self._gamma * next_critic_vals
        critic_loss = self._critic_loss(critic_vals, yi)

        # Actor loss
        critic_input = torch.cat((states, self._actor(states)), dim=1)
        actor_loss = -self._critic(critic_input).mean()

        # Update networks
        self._actor_optim.zero_grad()
        actor_loss.backward()
        self._actor_optim.step()

        self._critic_optim.zero_grad()
        critic_loss.backward()
        self._critic_optim.step()

        for target_p, actor_p in zip(self._actor_target.parameters(), self._actor.parameters()):
            target_p.data.copy_(actor_p.data * self._tau + target_p.data * (1.0 - self._tau))

        for target_p, critic_p in zip(self._critic_target.parameters(), self._critic.parameters()):
            target_p.data.copy_(critic_p.data * self._tau + target_p.data * (1.0 - self._tau))
