import pygame
import sys

from game2048 import Game2048

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

ACTION = {pygame.K_UP: 0, pygame.K_DOWN: 1, pygame.K_LEFT: 2, pygame.K_RIGHT: 3}


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


def main():
    screen = pygame.display.set_mode(WINDOW_SIZE)
    pygame.display.set_caption("2048")

    game = Game2048()
    draw_board(screen, game.board)

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key in ACTION:
                    _, _, done = game.step(ACTION[event.key])
                    draw_board(screen, game.board)
                    if done:
                        print("Game over!")
                        pygame.quit()
                        sys.exit()

if __name__ == "__main__":
    main()
