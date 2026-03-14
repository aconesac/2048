import numpy as np
import random
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim

from config import (
    GAMMA, EPSILON_START, EPSILON_MIN, EPSILON_DECAY,
    LEARNING_RATE, MEMORY_SIZE,
    HIDDEN_1, HIDDEN_2, LEAKY_RELU_ALPHA, DROPOUT_RATE, GRAD_CLIP_NORM,
)


class DQNNetwork(nn.Module):
    def __init__(self, state_size: int, action_size: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.BatchNorm1d(state_size),
            nn.Linear(state_size, HIDDEN_1),
            nn.LeakyReLU(LEAKY_RELU_ALPHA),
            nn.Dropout(DROPOUT_RATE),
            nn.Linear(HIDDEN_1, HIDDEN_2),
            nn.LeakyReLU(LEAKY_RELU_ALPHA),
            nn.Dropout(DROPOUT_RATE),
            nn.Linear(HIDDEN_2, action_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DQNAgent:
    def __init__(self, state_size: int, action_size: int):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=MEMORY_SIZE)

        self.gamma = GAMMA
        self.epsilon = EPSILON_START
        self.epsilon_min = EPSILON_MIN
        self.epsilon_decay = EPSILON_DECAY

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.main_model = DQNNetwork(state_size, action_size).to(self.device)
        self.target_model = DQNNetwork(state_size, action_size).to(self.device)
        self.target_model.load_state_dict(self.main_model.state_dict())
        self.target_model.eval()  # target never backprops

        self.optimizer = optim.Adam(self.main_model.parameters(), lr=LEARNING_RATE)
        self.loss_fn = nn.MSELoss()

    def preprocess_state(self, state: np.ndarray) -> np.ndarray:
        processed = np.zeros_like(state, dtype=np.float32)
        mask = state > 0
        processed[mask] = np.log2(state[mask])
        max_log = processed.max()
        if max_log > 0:
            processed /= max_log
        return processed

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((
            self.preprocess_state(state),
            action,
            reward,
            self.preprocess_state(next_state),
            done,
        ))

    def act(self, state: np.ndarray) -> int:
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        tensor = torch.FloatTensor(self.preprocess_state(state)).unsqueeze(0).to(self.device)
        self.main_model.eval()
        with torch.no_grad():
            q_values = self.main_model(tensor)
        self.main_model.train()
        return int(q_values.argmax(dim=1).item())

    def sample_memory(self, batch_size: int):
        idx = np.random.choice(len(self.memory), batch_size, replace=False)
        batch = [self.memory[i] for i in idx]
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states, dtype=np.float32),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32),
        )

    def train_step(self, states, actions, rewards, next_states, dones) -> float:
        states_t      = torch.FloatTensor(states).to(self.device)
        actions_t     = torch.LongTensor(actions).to(self.device)
        rewards_t     = torch.FloatTensor(rewards).to(self.device)
        next_states_t = torch.FloatTensor(next_states).to(self.device)
        dones_t       = torch.FloatTensor(dones).to(self.device)

        with torch.no_grad():
            max_next_q = self.target_model(next_states_t).max(dim=1).values
            target_q = rewards_t + (1.0 - dones_t) * self.gamma * max_next_q

        current_q = self.main_model(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)
        loss = self.loss_fn(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.main_model.parameters(), GRAD_CLIP_NORM)
        self.optimizer.step()
        return loss.item()

    def target_train(self):
        self.target_model.load_state_dict(self.main_model.state_dict())

    def reduce_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save(self, path: str):
        torch.save(self.main_model.state_dict(), path)

    def load(self, path: str):
        state_dict = torch.load(path, map_location=self.device)
        self.main_model.load_state_dict(state_dict)
        self.target_model.load_state_dict(state_dict)
        self.target_model.eval()
