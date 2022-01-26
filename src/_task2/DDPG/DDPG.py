import numpy as np
import random
import torch
from torch import nn, optim
from torch.autograd import Variable
from collections import deque


class FCNN(nn.Module):
    def __init__(self, layer_shapes=(5, 1), activation='linear'):
        super().__init__()
        # network layers
        self._layers = nn.ModuleList()
        for i in range(len(layer_shapes) - 1):
            layer = nn.Linear(layer_shapes[i], layer_shapes[i + 1], bias=True)
            self._layers.append(layer)

        # activation functions
        if activation == 'linear':
            self._activation = torch.nn.Identity()
        elif activation == 'tanh':
            self._activation = torch.nn.Tanh()
        else:
            raise Exception('activation function %s not recognized' % activation)

    def copy_weights(self, other):
        for param, param_value in zip(self.parameters(), other.parameters()):
            param.data.copy_(param_value.data)
        return self

    def forward(self, x):
        # Forward pass through the network
        for layer in self._layers[:-1]:  # Intermediate layers
            x = torch.sigmoid(layer(x))
        return self._activation(self._layers[-1](x))  # Output layer


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
        if len(self._buffer) < self._batch_size:
            experience = self._buffer
        else:
            experience = random.sample(self._buffer, k=self._batch_size - 1)
            experience.append(self._buffer[-1])  # Always learn from last experience!

        states, actions, rewards, next_states = zip(*experience)
        states = torch.Tensor(np.array(states))
        actions = torch.Tensor(np.array(actions))
        rewards = torch.Tensor(np.array(rewards))
        next_states = torch.Tensor(np.array(next_states))
        return states, actions, rewards, next_states


class DDPGAgent:
    def __init__(self, layer_shapes=(5, 2), actor_lrate=1e-4, critic_lrate=1e-3,
                 gamma=0.9, tau=1e-2, max_replay_buffer_size=2048, replay_size=64):
        # Hyper-parameters
        self._gamma = gamma
        self._tau = tau

        # Store experience in replay buffer
        self._replay_buffer = ReplayBuffer(max_replay_buffer_size, replay_size)

        # Actor, Critic and Target networks
        self._actor = FCNN(layer_shapes, activation='tanh')
        self._actor_target = FCNN(layer_shapes, activation='tanh')
        self._actor_target.copy_weights(self._actor)

        critic_shape = (layer_shapes[0] + layer_shapes[-1],) + layer_shapes[1:-1] + (1,)
        self._critic = FCNN(critic_shape, activation='linear')
        self._critic_target = FCNN(critic_shape, activation='linear')
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

        # Weigh target networks
        for target_p, actor_p in zip(self._actor_target.parameters(), self._actor.parameters()):
            target_p.data.copy_(actor_p.data * self._tau + target_p.data * (1.0 - self._tau))

        for target_p, critic_p in zip(self._critic_target.parameters(), self._critic.parameters()):
            target_p.data.copy_(critic_p.data * self._tau + target_p.data * (1.0 - self._tau))
