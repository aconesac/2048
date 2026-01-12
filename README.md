# 2048 Deep Q-Learning Agent

A Deep Reinforcement Learning agent that learns to play the 2048 game using Deep Q-Networks (DQN) with TensorFlow/Keras.

## 🎮 Overview

This project implements a DQN agent trained to master the 2048 puzzle game through self-play. The agent learns optimal tile merging strategies by exploring different moves and receiving rewards based on game progression.

## 🧠 Features

- **Deep Q-Network (DQN)** with experience replay and target networks
- **Large neural architecture**: 512→256→128→64→32 neurons with LeakyReLU activation
- **Smart state encoding**: Logarithmic tile representation with normalization
- **Custom reward shaping**: 
  - Penalties for invalid moves and game over
  - Bonuses for creating empty spaces and achieving new max tiles
  - Normalized score progression rewards
- **Training optimizations**:
  - TensorFlow XLA JIT compilation
  - Multi-threaded CPU utilization
  - Experience replay buffer (10,000 transitions)
  - Epsilon-greedy exploration with decay
- **Progress tracking**: Real-time training metrics with tqdm progress bars
- **Visualization**: Automated plot generation for scores, losses, and rewards

## 📁 Project Structure

```
2048/
├── RLAgent.py          # DQN agent implementation
├── game2048.py         # 2048 game logic and reward calculation
├── enviroment.py       # Training loop and orchestration
├── gameInterface.py    # Pygame visualization interface
├── 2048.py            # Play game manually or with trained agent
├── test.py            # Testing utilities
└── results/           # Training plots and saved models
```

## 🚀 Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Train the Agent

```bash
python enviroment.py
```

Training runs for 10,800 episodes by default with the following hyperparameters:
- Learning rate: 0.0001
- Discount factor (γ): 0.95
- Epsilon decay: 0.9995
- Batch size: 128
- Training frequency: Every 8 steps

### Play Manually

```bash
python 2048.py
```

Use arrow keys to control the game.

## 🎯 Performance

The agent learns to:
- ✅ Consistently achieve **256 tiles**
- 🎯 Reach **512 tiles** with proper training
- 📈 Average score: 200-300+ per game
- 🧩 Develop emergent strategies like corner-focused play

## 🛠️ Key Components

### State Representation
- 16-dimensional vector (flattened 4×4 board)
- Log₂ encoding: `log2(tile_value)` for non-zero tiles
- Normalized by log₂(2048) to [0, 1] range

### Neural Network Architecture
```python
Input (16) 
→ BatchNorm 
→ Dense(512, LeakyReLU) + Dropout(0.3)
→ Dense(256, LeakyReLU) + Dropout(0.3)
→ Dense(128, LeakyReLU) + Dropout(0.2)
→ Dense(64, LeakyReLU)
→ Dense(32, LeakyReLU)
→ Output(4, Linear)  # Q-values for [up, down, left, right]
```

### Reward Function
```python
- Game over: -10.0
- Invalid move: -5.0
- Creating empty space: +5.0 + normalized_change
- New max tile: bonus scaled by log₂ ratio
- Valid move: normalized board sum change
```

## 📊 Training Outputs

The training process generates:
- `model-{timestamp}.keras` - Trained model weights
- `scores-{timestamp}.png` - Episode scores over time
- `losses-{timestamp}.png` - Training loss curve
- `rewards-{timestamp}.png` - Reward progression

## 🔧 Configuration

Edit hyperparameters in `enviroment.py`:
```python
episodes = 10800           # Total training episodes
training_freq = 8          # Train every N steps
num_train_cycles = 3       # Training iterations per trigger
```

Edit agent parameters in `RLAgent.py`:
```python
epsilon_decay = 0.9995     # Exploration decay rate
gamma = 0.95               # Discount factor
learning_rate = 0.0001     # Adam optimizer learning rate
```

## 🤝 Contributing

Contributions welcome! Some ideas for improvement:
- [ ] Implement Dueling DQN architecture
- [ ] Add Prioritized Experience Replay (PER)
- [ ] Try n-step returns
- [ ] Experiment with different reward structures
- [ ] Add model checkpointing and early stopping
- [ ] Implement convolutional layers for spatial patterns

## 📝 License

MIT License - feel free to use for learning and experimentation!

## 🙏 Acknowledgments

Built with TensorFlow, Keras, and Pygame. Inspired by DeepMind's DQN paper and the classic 2048 game.
