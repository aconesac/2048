import numpy as np
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.set_num_threads(1)  # tiny MLP: extra threads only add overhead and block parallel runs

from config import (
    GAMMA, EPSILON_START, EPSILON_MIN, EPSILON_DECAY,
    LEARNING_RATE, MEMORY_SIZE, REWARD_SCALE, HIDDEN_1, HIDDEN_2, GRAD_CLIP_NORM, TAU,
)

MAX_EXP = 16  # tiles up to 2^15 are representable


class ValueNetwork(nn.Module):
    """MLP over the one-hot encoded board -> scalar afterstate value."""

    def __init__(self, state_size: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_size * MAX_EXP, HIDDEN_1), nn.ReLU(),
            nn.Linear(HIDDEN_1, HIDDEN_2), nn.ReLU(),
            nn.Linear(HIDDEN_2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class ReplayBuffer:
    """Stores afterstate transitions. For the next state we keep all 4 candidate
    afterstates, their merge rewards and validity so the TD target can be computed
    without re-simulating the game."""

    def __init__(self, capacity: int, state_size: int, action_size: int):
        self.capacity = capacity
        self.afters = np.zeros((capacity, state_size), dtype=np.int8)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self.next_afters = np.zeros((capacity, action_size, state_size), dtype=np.int8)
        self.next_rewards = np.zeros((capacity, action_size), dtype=np.float32)
        self.next_valid = np.zeros((capacity, action_size), dtype=bool)
        self.pos = 0
        self.size = 0

    def __len__(self):
        return self.size

    def add(self, after, reward, done, next_afters, next_rewards, next_valid):
        i = self.pos
        self.afters[i], self.rewards[i], self.dones[i] = after, reward, done
        self.next_afters[i], self.next_rewards[i], self.next_valid[i] = next_afters, next_rewards, next_valid
        self.pos = (i + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int):
        idx = np.random.randint(0, self.size, batch_size)
        return (self.afters[idx], self.rewards[idx], self.dones[idx],
                self.next_afters[idx], self.next_rewards[idx], self.next_valid[idx])


class AfterstateAgent:
    """TD learning of afterstate values V(after) with a target network.

    Action selection:  argmax_a  r(s, a) + gamma * V(after(s, a))
    TD target:         r + gamma * max_a' [ r(s', a') + gamma * V_target(after(s', a')) ]
    """

    def __init__(self, state_size: int = 16, action_size: int = 4):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = ReplayBuffer(MEMORY_SIZE, state_size, action_size)

        self.gamma = GAMMA
        self.epsilon = EPSILON_START
        self.epsilon_min = EPSILON_MIN
        self.epsilon_decay = EPSILON_DECAY

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ValueNetwork(state_size).to(self.device)
        self.target_model = ValueNetwork(state_size).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()
        self.optimizer = optim.Adam(self.model.parameters(), lr=LEARNING_RATE)

    @staticmethod
    def exponents(boards: np.ndarray) -> np.ndarray:
        """Tile values -> log2 exponents (0 for empty)."""
        boards = np.asarray(boards)
        out = np.zeros(boards.shape, dtype=np.int8)
        mask = boards > 0
        out[mask] = np.log2(boards[mask]).astype(np.int8)
        return out

    @staticmethod
    def _encode(exps: torch.Tensor) -> torch.Tensor:
        return F.one_hot(exps.long(), MAX_EXP).float().flatten(start_dim=-2)

    def action_values(self, afters: np.ndarray, gains: np.ndarray) -> np.ndarray:
        """r + gamma * V(after) for each of the 4 candidate afterstates."""
        exps = torch.from_numpy(self.exponents(afters)).to(self.device)
        with torch.no_grad():
            v = self.model(self._encode(exps)).cpu().numpy()
        return gains / REWARD_SCALE + self.gamma * v

    def act(self, afters: np.ndarray, gains: np.ndarray, valid: np.ndarray, greedy: bool = False) -> int:
        """Epsilon-greedy over the *valid* moves only."""
        valid_idx = np.flatnonzero(valid)
        if not greedy and np.random.rand() < self.epsilon:
            return int(random.choice(valid_idx))
        q = self.action_values(afters[valid_idx], gains[valid_idx])
        return int(valid_idx[np.argmax(q)])

    def remember(self, after, reward, done, next_afters, next_rewards, next_valid):
        self.memory.add(self.exponents(after), reward / REWARD_SCALE, done,
                        self.exponents(next_afters), next_rewards / REWARD_SCALE, next_valid)

    def train_step(self, batch_size: int) -> float:
        a, r, d, na, nr, nv = self.memory.sample(batch_size)
        dev = self.device
        a = self._encode(torch.from_numpy(a).to(dev))
        r = torch.from_numpy(r).to(dev)
        d = torch.from_numpy(d).to(dev)
        nr = torch.from_numpy(nr).to(dev)
        nv = torch.from_numpy(nv).to(dev)
        na = self._encode(torch.from_numpy(na).to(dev))          # (B, 4, 256)

        with torch.no_grad():
            nv_values = self.target_model(na.flatten(0, 1)).view(batch_size, -1)
            options = (nr + self.gamma * nv_values).masked_fill(~nv, -1e9)
            best = options.max(dim=1).values
            target = r + (1.0 - d) * self.gamma * best

        loss = F.smooth_l1_loss(self.model(a), target)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), GRAD_CLIP_NORM)
        self.optimizer.step()
        self._soft_update()
        return loss.item()

    def _soft_update(self):
        with torch.no_grad():
            for t, m in zip(self.target_model.parameters(), self.model.parameters()):
                t.mul_(1 - TAU).add_(m, alpha=TAU)

    def reduce_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, path: str):
        torch.save(self.model.state_dict(), path)

    def load(self, path: str):
        state_dict = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.target_model.load_state_dict(state_dict)
        self.target_model.eval()
