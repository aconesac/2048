# Reward: merge score / REWARD_SCALE
REWARD_SCALE = 100.0
GAME_OVER_PENALTY = 0.0

# Agent hyperparameters
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_MIN = 0.02
EPSILON_DECAY = 0.998       # per episode
LEARNING_RATE = 0.0005
MEMORY_SIZE = 100000
BATCH_SIZE = 128
GRAD_CLIP_NORM = 10.0
TAU = 0.01                  # soft target update rate

# Network architecture
HIDDEN_1 = 512
HIDDEN_2 = 256

# Training loop
EPISODES = 3000
TRAINING_FREQ = 4           # env steps between gradient steps
MIN_REPLAY = 2000
