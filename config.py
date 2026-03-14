# Game / Reward constants (moved from game2048.py module-level)
LAMBDA = 5.0
MAX_EXPECTED_TILE = 2048
INVALID_MOVE_PENALTY = -5.0
GAME_OVER_PENALTY = -10.0
MERGE_REWARD_SCALE = 10000
NEW_MAX_SCALE = 5.0

# Agent hyperparameters (moved from RLAgent.py __init__)
GAMMA = 0.95
EPSILON_START = 1.0
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.9995
LEARNING_RATE = 0.0001
MEMORY_SIZE = 10000
BATCH_SIZE = 128
GRAD_CLIP_NORM = 1.0

# Network architecture (extracted from _build_big_model)
HIDDEN_1 = 256
HIDDEN_2 = 128
LEAKY_RELU_ALPHA = 0.01
DROPOUT_RATE = 0.1

# Training loop (moved from environment.py top-of-__main__)
EPISODES = 2000
TRAINING_FREQ = 8
NUM_TRAIN_CYCLES = 3
TARGET_SYNC_FREQ = 100
