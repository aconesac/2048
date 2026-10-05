"""Watch a trained agent play 2048 in a pygame window.

Usage: python watch.py model-<timestamp>.pt [--delay 0.15]
Keys: SPACE pause/resume, UP/DOWN faster/slower, R new game, ESC or close the window to quit.
"""
import argparse
import sys
import time

import pygame

from game2048 import Game2048
from gameInterface import gameInterface
from RLAgent import AfterstateAgent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("--delay", type=float, default=0.15, help="seconds between moves")
    args = parser.parse_args()

    agent = AfterstateAgent()
    agent.load(args.model)

    env = Game2048()
    interface = gameInterface(env, draw=True)
    delay, paused, games, moves = args.delay, False, 0, 0

    def caption():
        state = " [PAUSED]" if paused else ""
        pygame.display.set_caption(f"2048 agent - game {games + 1}, moves {moves}, delay {delay:.2f}s{state}")

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_UP:
                    delay = max(0.0, delay / 1.5)
                elif event.key == pygame.K_DOWN:
                    delay = min(2.0, max(delay, 0.02) * 1.5)
                elif event.key == pygame.K_r:
                    env, moves = Game2048(), 0
                    interface.setEnv(env)
        caption()
        if paused:
            time.sleep(0.05)
            continue

        afters, gains, valid = env.afterstates()
        _, _, done = env.step(agent.act(afters, gains, valid, greedy=True))
        moves += 1
        interface.setEnv(env)
        time.sleep(delay)

        if done:
            print(f"Game {games + 1} over: max tile {env.board.max()}, board sum {env.board.sum()}, {moves} moves")
            games += 1
            time.sleep(2)
            env, moves = Game2048(), 0
            interface.setEnv(env)


if __name__ == "__main__":
    main()
