# 2048 Deep Q-Learning Agent

A Deep Reinforcement Learning agent that learns to play the 2048 game using afterstate value learning (a DQN-style TD method) with PyTorch.

## 🎮 Overview

This project implements a DQN agent trained to master the 2048 puzzle game through self-play. The agent learns optimal tile merging strategies by exploring different moves and receiving rewards based on game progression.

## 🧠 Method

- **Afterstate TD learning**: the network estimates the value `V` of the board right after a move (before the random tile spawns), which is deterministic in 2048. Action = `argmax_a r(s,a) + γ·V(after(s,a))`; target = `r + γ·max_a' [r(s',a') + γ·V_target(after(s',a'))]`.
- **One-hot board encoding** (16 cells × 16 exponents) instead of a state-dependent normalisation.
- **Invalid moves are masked**, so no steps are wasted on them.
- Target network with soft updates, Huber loss, replay buffer (100k), epsilon-greedy exploration (per-episode decay).
- **Symmetry**: the value of a board equals that of its 8 rotations/reflections, so training batches are randomly symmetrised and the value is averaged over the 8 views when acting.
- Reward = merge score / 100.
- Fast game engine with cached row transitions.

## 📁 Project Structure

```
2048/
├── RLAgent.py       # Afterstate agent, network and replay buffer
├── game2048.py      # Game logic
├── config.py        # Hyperparameters
├── train.py         # Training loop (python train.py [episodes])
├── evaluate.py      # Greedy evaluation (python evaluate.py model.pt)
├── gameInterface.py # Pygame visualization
└── 2048.py          # Play manually
```

## 🚀 Quick Start

```bash
pip install -r requirements.txt
python train.py 3000        # saves the best checkpoint as model-<timestamp>.pt
python evaluate.py model-<timestamp>.pt --games 200
python 2048.py              # play manually with the arrow keys
```

Hyperparameters live in `config.py`.

## 🎯 Performance

Greedy play, average over 200 games (board sum = sum of the final tiles):

| | Avg board sum | Avg max tile | Games reaching ≥2048 |
|---|---|---|---|
| Previous DQN (2000 episodes) | 250 | 103 | 0% |
| Afterstate agent, no symmetry (3000 episodes) | 767 | 386 | 0% (≥512: 48%) |
| Afterstate agent + symmetry (3000 episodes) | 2616 | 1530 | 50.5% |
