import random
import numpy as np

from config import GAME_OVER_PENALTY, REWARD_SCALE

# Row transitions are cached: a row is a tuple of 4 tile values, the cache maps it to
# (row after sliding left, score gained from merges).
_LEFT_CACHE: dict = {}


def _slide_left(row: tuple) -> tuple:
    cached = _LEFT_CACHE.get(row)
    if cached is not None:
        return cached
    tiles = [v for v in row if v]
    out, gain, i = [], 0, 0
    while i < len(tiles):
        if i + 1 < len(tiles) and tiles[i] == tiles[i + 1]:
            out.append(tiles[i] * 2)
            gain += tiles[i] * 2
            i += 2
        else:
            out.append(tiles[i])
            i += 1
    out += [0] * (len(row) - len(out))
    result = (tuple(out), gain)
    _LEFT_CACHE[row] = result
    return result


class Game2048:
    """Actions: 0=up, 1=down, 2=left, 3=right."""

    def __init__(self):
        self.board = np.zeros((4, 4), dtype=int)
        self.action_space = 4
        self.add_new_tile(self.board)
        self.add_new_tile(self.board)

    def add_new_tile(self, board: np.ndarray) -> None:
        empty = np.flatnonzero(board == 0)
        if empty.size:
            board.flat[random.choice(empty)] = 2 if random.random() < 0.9 else 4

    def _move(self, board: np.ndarray, action: int):
        """Slide without spawning a tile. Returns (new_board, merge_score)."""
        # Rotate so every move becomes a "slide left".
        if action == 0:
            view = board.T
        elif action == 1:
            view = board.T[:, ::-1]
        elif action == 2:
            view = board
        else:
            view = board[:, ::-1]
        gain = 0
        rows = []
        for row in view.tolist():
            new_row, g = _slide_left(tuple(row))
            rows.append(new_row)
            gain += g
        out = np.array(rows, dtype=int)
        if action == 0:
            out = out.T
        elif action == 1:
            out = out[:, ::-1].T
        elif action == 3:
            out = out[:, ::-1]
        return out, gain

    def valid_moves(self, board: np.ndarray = None) -> np.ndarray:
        board = self.board if board is None else board
        return np.array(
            [not np.array_equal(self._move(board, a)[0], board) for a in range(4)]
        )

    def afterstates(self, board: np.ndarray = None):
        """For each action: (board after sliding but before the random spawn, merge score, valid)."""
        board = self.board if board is None else board
        afters = np.zeros((4, 16), dtype=int)
        gains = np.zeros(4, dtype=np.float32)
        valid = np.zeros(4, dtype=bool)
        for a in range(4):
            new_board, gain = self._move(board, a)
            if not np.array_equal(new_board, board):
                afters[a] = new_board.flatten()
                gains[a] = gain
                valid[a] = True
        return afters, gains, valid

    def game_over(self, board: np.ndarray) -> bool:
        return not self.valid_moves(board).any()

    def get_state(self) -> np.ndarray:
        return self.board.flatten()

    def step(self, action: int) -> tuple[np.ndarray, float, bool]:
        new_board, gain = self._move(self.board, action)
        if np.array_equal(new_board, self.board):
            # Invalid move: board unchanged, no reward, game continues.
            return self.get_state(), 0.0, False
        self.board = new_board
        self.add_new_tile(self.board)
        done = self.game_over(self.board)
        reward = gain / REWARD_SCALE + (GAME_OVER_PENALTY if done else 0.0)
        return self.get_state(), reward, done
