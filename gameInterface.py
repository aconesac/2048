import pygame
import sys

class gameInterface:
    def __init__(self, env, draw=True):
        self.draw = draw
        self.grid_size = 4
        self.tile_size = 100
        self.tile_margin = 10
        self.background_color = (187, 173, 160)
        self.font_color = (119, 110, 101)
        self.font_size = 55
        self.size = (self.grid_size * self.tile_size + (self.grid_size + 1) * self.tile_margin,
                     self.grid_size * self.tile_size + (self.grid_size + 1) * self.tile_margin)
        self.colors = {
            0: (255, 255, 255),
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
            2048: (237, 194, 46),
        }
        
        
        self.env = env
        if self.draw:
            pygame.init()
            self.font = pygame.font.Font(None, 36)
            self.screen = pygame.display.set_mode(self.size)
            pygame.display.set_caption("2048")
            self.draw_board()
        
    def draw_board(self):
        self.screen.fill(self.background_color)
        for row in range(self.grid_size):
            for col in range(self.grid_size):
                value = self.env.board[row, col]
                color = self.colors.get(value, (60, 58, 50))
                rect = pygame.Rect(col * self.tile_size + (col + 1) * self.tile_margin, 
                                   row * self.tile_size + (row + 1) * self.tile_margin, 
                                   self.tile_size, self.tile_size)
                pygame.draw.rect(self.screen, color, rect)
                if value:
                    text = self.font.render(str(value), True, self.font_color)
                    text_rect = text.get_rect(center=rect.center)
                    self.screen.blit(text, text_rect)
        pygame.display.update()
        
    def setEnv(self, env):
        self.env = env
        if self.draw:
            self.draw_board()
        