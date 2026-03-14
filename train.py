import os
import sys
import time

import pygame
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from game2048 import Game2048
from RLAgent import DQNAgent
from gameInterface import gameInterface
from config import EPISODES, TRAINING_FREQ, NUM_TRAIN_CYCLES, TARGET_SYNC_FREQ, BATCH_SIZE

if __name__ == "__main__":
    env = Game2048()
    agent = DQNAgent(env.get_state().shape[0], env.action_space)
    # agent.load("model-<timestamp>.pt")  # uncomment to resume

    interface = gameInterface(env, draw=False)

    scores, losses, rewards_track = [], [], []

    for episode in tqdm(range(EPISODES), desc="Episodes"):
        env = Game2048()
        state = env.get_state()
        done = False
        step = 0

        while not done:
            if interface.draw:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        sys.exit()

            action = agent.act(state)
            next_state, reward, done = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            interface.setEnv(env)
            rewards_track.append(reward)

            if len(agent.memory) > BATCH_SIZE and step % TRAINING_FREQ == 0:
                for _ in range(NUM_TRAIN_CYCLES):
                    loss = agent.train_step(*agent.sample_memory(BATCH_SIZE))
                    losses.append(loss)

            if step % TARGET_SYNC_FREQ == 0:
                agent.target_train()

            step += 1

        scores.append([env.board.sum(), env.board.max()])
        agent.reduce_epsilon()

        if (episode + 1) % 100 == 0:
            recent = np.array(scores[-100:])
            print(
                f"\n[Episode {episode+1}/{EPISODES}] "
                f"Avg Score: {recent[:, 0].mean():.1f}, "
                f"Avg Max Tile: {recent[:, 1].mean():.1f}, "
                f"Best Max Tile: {recent[:, 1].max():.0f}, "
                f"Epsilon: {agent.epsilon:.4f}"
            )

    date = time.strftime("%Y-%m-%d_%H-%M-%S")
    agent.save(f"model-{date}.pt")

    scores = np.array(scores)
    os.makedirs("results", exist_ok=True)

    plt.figure()
    plt.plot(scores, label=["Score", "Max Tile"])
    plt.legend()
    plt.savefig(f"results/scores-{date}.png")

    plt.figure()
    plt.plot(losses)
    plt.title("Training Loss")
    plt.savefig(f"results/losses-{date}.png")

    plt.figure()
    plt.plot(rewards_track)
    plt.title("Reward per Step")
    plt.savefig(f"results/rewards-{date}.png")

    pygame.quit()
