import random
import numpy as np

LAMBDA = 5.0
# Constants for reward scaling
MAX_EXPECTED_TILE = 2048  # Highest expected tile value
INVALID_MOVE_PENALTY = -5.0  # Scaled penalty for invalid moves
GAME_OVER_PENALTY = -10.0  # Scaled penalty for game over
MERGE_REWARD_SCALE = 10000  # Scale factor for merge rewards
NEW_MAX_SCALE = 5.0  # Scale factor for new max tile bonus
class Game2048:
    def __init__(self):
        self.board = self.init_board()
        self.action_space = 4
    
    def init_board(self):
        board = np.zeros((4, 4), dtype=int)
        self.add_new_tile(board)
        self.add_new_tile(board)
        return board

    def add_new_tile(self, board: np.ndarray) -> None:
        empty_cells = list(zip(*np.where(board == 0)))
        if empty_cells:
            row, col = random.choice(empty_cells)
            board[row, col] = 2 if random.random() < 0.9 else 4

    def slide_left(self, row: np.ndarray) -> np.ndarray:
        new_row = [num for num in row if num != 0]
        for i in range(len(new_row) - 1):
            if new_row[i] == new_row[i + 1]:
                new_row[i] *= 2
                new_row[i + 1] = 0
        new_row = [num for num in new_row if num != 0]
        new_row += [0] * (len(row) - len(new_row))
        return new_row

    def move_left(self, board: np.ndarray) -> np.ndarray:
        new_board = np.array([self.slide_left(row) for row in board])
        if not np.array_equal(board, new_board):
            self.add_new_tile(new_board)
            return new_board
        return board  # Return original if no change

    def move_right(self, board: np.ndarray) -> np.ndarray:
        new_board = np.array([self.slide_left(row[::-1])[::-1] for row in board])
        if not np.array_equal(board, new_board):
            self.add_new_tile(new_board)
            return new_board
        return board  # Return original if no change

    def move_up(self, board: np.ndarray) -> np.ndarray:
        new_board = np.array([self.slide_left(row) for row in board.T]).T
        if not np.array_equal(board, new_board):
            self.add_new_tile(new_board)
            return new_board    
        return board  # Return original if no change
    
    def move_down(self, board: np.ndarray) -> np.ndarray:
        new_board = np.array([self.slide_left(row[::-1])[::-1] for row in board.T]).T
        if not np.array_equal(board, new_board):
            self.add_new_tile(new_board)
            return new_board
        return board  # Return original if no change
    
    def game_over(self, board: np.ndarray):
        """
        Check if the game is over by checking if there are any empty cells or if there are any adjacent cells with the same
        value in the same row or column.    
        
        """
        if np.any(board == 0):
            return False
        for row in board:
            for i in range(len(row) - 1):
                if row[i] == row[i + 1]:
                    return False
        for col in board.T:
            for i in range(len(col) - 1):
                if col[i] == col[i + 1]:
                    return False
        return True

    def get_state(self) -> np.ndarray:
        return self.board.flatten()
    
    def calculate_reward(self, old_board, new_board, done):
        # Game over penalty
        if done:
            return GAME_OVER_PENALTY
        
        # Invalid move penalty
        if np.array_equal(old_board, new_board):
            return INVALID_MOVE_PENALTY
        
        # Base reward: normalized change in board sum
        sum_change = np.sum(new_board) - np.sum(old_board)
        normalized_change = sum_change / MAX_EXPECTED_TILE
        
        # Bonus for new max tile
        if new_board.max() > old_board.max():
            max_tile_bonus = NEW_MAX_SCALE * (np.log2(new_board.max()) / np.log2(MAX_EXPECTED_TILE))
            return normalized_change + max_tile_bonus
        
        # Bonus for merging tiles
        if np.sum(np.where(new_board == 0, 1, 0)) > np.sum(np.where(old_board == 0, 1, 0)):
            return normalized_change + LAMBDA
        
        # Regular merge reward
        return normalized_change

    def step(self, action: int) -> tuple[np.ndarray, int, bool]:
        old_board = np.copy(self.board) # Save the old board to check if the board has changed
        
        # Perform the action update the board
        if action == 0:
            self.board = self.move_up(self.board)
        elif action == 1:
            self.board = self.move_down(self.board)
        elif action == 2:
            self.board = self.move_left(self.board)
        elif action == 3:
            self.board = self.move_right(self.board)
            
        done = self.game_over(self.board)
        
        reward = self.calculate_reward(old_board, self.board, done)
            
        
        return self.get_state(), reward, done
