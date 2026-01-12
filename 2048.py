import pygame
import numpy as np
import random
import sys

# Constants
GRID_SIZE = 4
TILE_SIZE = 100
TILE_MARGIN = 10
WINDOW_SIZE = (GRID_SIZE * TILE_SIZE + (GRID_SIZE + 1) * TILE_MARGIN,
               GRID_SIZE * TILE_SIZE + (GRID_SIZE + 1) * TILE_MARGIN)
BACKGROUND_COLOR = (187, 173, 160)
TILE_COLORS = {
    0: (205, 193, 180),
    2: (238, 228, 218),
    4: (237, 224, 200),
    8: (242, 177, 121),
    16: (245, 149, 99),
    32: (246, 124, 95),
    64: (246, 94, 59),
    128: (237, 207, 114),
    256: (237, 204, 97),
    512: (237, 200, 80),
    1024: (237, 197, 63),
    2048: (237, 194, 46)
}
FONT_COLOR = (119, 110, 101)
FONT_SIZE = 55

pygame.init()
font = pygame.font.Font(None, FONT_SIZE)

def init_board():
    board = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
    add_new_tile(board)
    add_new_tile(board)
    return board

def add_new_tile(board):
    empty_cells = list(zip(*np.where(board == 0)))
    if empty_cells:
        row, col = random.choice(empty_cells)
        board[row, col] = 2 if random.random() < 0.9 else 4

def draw_board(screen, board):
    screen.fill(BACKGROUND_COLOR)
    for row in range(GRID_SIZE):
        for col in range(GRID_SIZE):
            value = board[row, col]
            color = TILE_COLORS.get(value, (60, 58, 50))
            rect = pygame.Rect(col * TILE_SIZE + (col + 1) * TILE_MARGIN,
                               row * TILE_SIZE + (row + 1) * TILE_MARGIN,
                               TILE_SIZE, TILE_SIZE)
            pygame.draw.rect(screen, color, rect)
            if value != 0:
                text = font.render(str(value), True, FONT_COLOR)
                text_rect = text.get_rect(center=rect.center)
                screen.blit(text, text_rect)
    pygame.display.update()

def slide_left(row):
    new_row = [num for num in row if num != 0]
    for i in range(len(new_row) - 1):
        if new_row[i] == new_row[i + 1]:
            new_row[i] *= 2
            new_row[i + 1] = 0
    new_row = [num for num in new_row if num != 0]
    new_row += [0] * (len(row) - len(new_row))
    return new_row

def move_left(board):
    new_board = np.array([slide_left(row) for row in board])
    if not np.array_equal(board, new_board):
        add_new_tile(new_board)
    return new_board

def move_right(board):
    new_board = np.array([slide_left(row[::-1])[::-1] for row in board])
    if not np.array_equal(board, new_board):
        add_new_tile(new_board)
    return new_board

def move_up(board):
    new_board = np.array([slide_left(row) for row in board.T]).T
    if not np.array_equal(board, new_board):
        add_new_tile(new_board)
    return new_board

def move_down(board):
    new_board = np.array([slide_left(row[::-1])[::-1] for row in board.T]).T
    if not np.array_equal(board, new_board):
        add_new_tile(new_board)
    return new_board

def game_over(board):
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

def main():
    screen = pygame.display.set_mode(WINDOW_SIZE)
    pygame.display.set_caption("2048")

    board = init_board()
    draw_board(screen, board)

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    board = move_up(board)
                elif event.key == pygame.K_DOWN:
                    board = move_down(board)
                elif event.key == pygame.K_LEFT:
                    board = move_left(board)
                elif event.key == pygame.K_RIGHT:
                    board = move_right(board)

                draw_board(screen, board)

                if game_over(board):
                    print("Game over!")
                    pygame.quit()
                    sys.exit()

if __name__ == "__main__":
    main()
