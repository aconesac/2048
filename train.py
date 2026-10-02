import os
import sys
import time

import numpy as np
from tqdm import tqdm
import matplotlib

from game2048 import Game2048
from RLAgent import AfterstateAgent
from evaluate import evaluate, summary
from config import EPISODES, TRAINING_FREQ, MIN_REPLAY, BATCH_SIZE

EVAL_EVERY = 500

if __name__ == "__main__":
    episodes = int(sys.argv[1]) if len(sys.argv) > 1 else EPISODES
    agent = AfterstateAgent()
    # agent.load("model-<timestamp>.pt")  # uncomment to resume

    scores, losses = [], []
    best_eval, best_state = -1, None
    date = time.strftime("%Y-%m-%d_%H-%M-%S")
    os.makedirs("results", exist_ok=True)

    for episode in tqdm(range(episodes), desc="Episodes"):
        env = Game2048()
        afters, gains, valid = env.afterstates()
        done, step = False, 0
        while not done:
            action = agent.act(afters, gains, valid)
            after, gain = afters[action], gains[action]
            _, _, done = env.step(action)
            if done:
                afters, gains, valid = np.zeros_like(afters), np.zeros_like(gains), np.zeros_like(valid)
            else:
                afters, gains, valid = env.afterstates()
            agent.remember(after, gain, done, afters, gains, valid)
            if len(agent.memory) >= MIN_REPLAY and step % TRAINING_FREQ == 0:
                losses.append(agent.train_step(BATCH_SIZE))
            step += 1

        scores.append([env.board.sum(), env.board.max()])
        agent.reduce_epsilon()

        if (episode + 1) % 100 == 0:
            recent = np.array(scores[-100:])
            print(f"\n[Episode {episode+1}/{episodes}] Avg Score: {recent[:, 0].mean():.1f}, "
                  f"Avg Max Tile: {recent[:, 1].mean():.1f}, Best Max Tile: {recent[:, 1].max():.0f}, "
                  f"Epsilon: {agent.epsilon:.3f}", flush=True)

        if (episode + 1) % EVAL_EVERY == 0:
            ev = evaluate(agent, 50)
            print("  greedy eval:", summary(*ev), flush=True)
            if ev[0].mean() > best_eval:  # keep the best checkpoint
                best_eval = ev[0].mean()
                agent.save(f"model-{date}.pt")

    print("Best checkpoint saved to", f"model-{date}.pt")

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    scores = np.array(scores)
    plt.figure(); plt.plot(scores[:, 0], label="Score"); plt.plot(scores[:, 1], label="Max Tile")
    plt.legend(); plt.savefig(f"results/scores-{date}.png")
    plt.figure(); plt.plot(losses); plt.title("Training Loss"); plt.savefig(f"results/losses-{date}.png")
