import argparse
import numpy as np

from game2048 import Game2048
from RLAgent import AfterstateAgent


def evaluate(agent: AfterstateAgent, games: int = 100):
    """Greedy play. Returns (scores, max_tiles) as arrays."""
    scores, tiles = [], []
    for _ in range(games):
        env = Game2048()
        done = False
        while not done:
            afters, gains, valid = env.afterstates()
            _, _, done = env.step(agent.act(afters, gains, valid, greedy=True))
        scores.append(env.board.sum())
        tiles.append(env.board.max())
    return np.array(scores), np.array(tiles)


def summary(scores, tiles) -> str:
    vals, cnt = np.unique(tiles, return_counts=True)
    dist = ", ".join(f"{v}:{c}" for v, c in zip(vals, cnt))
    return f"avg score {scores.mean():.0f}, avg max tile {tiles.mean():.0f} [{dist}]"


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("model")
    p.add_argument("--games", type=int, default=100)
    args = p.parse_args()
    agent = AfterstateAgent()
    agent.load(args.model)
    print(summary(*evaluate(agent, args.games)))
