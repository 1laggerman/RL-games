from src.base import Player, Game

import pygame

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREY = (100, 100, 100)

class pygamePlayer(Player):
    def __init__(self, game_board: Game, name: str) -> None:
        super().__init__(game_board, name)
        window_len = 800
        SCREEN_WIDTH = window_len
        SCREEN_HEIGHT = window_len

        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        # clock = pygame.time.Clock()

    def get_move(self):
        self.screen.fill((0, 0, 0))
        step = window_len / 3

        pygame.draw.line(self.screen, (255, 255, 255), (0, step), (window_len, step), 10)
        pygame.draw.line(self.screen, (255, 255, 255), (0, step * 2), (window_len, step * 2), 10)

        pygame.draw.line(self.screen, (255, 255, 255), (step, 0), (step, window_len), 10)
        pygame.draw.line(self.screen, (255, 255, 255), (step * 2, 0), (step * 2, window_len), 10)
        
        for piece in self.game.roles[0].pieces:
            draw_X(self.screen, piece.location, step)

        for piece in self.game.roles[1].pieces:
            draw_O(self.screen, piece.location, step)

        if self.game.state != gameState.ONGOING:
            size = (SCREEN_WIDTH - 100, SCREEN_HEIGHT // 3)
            pos = ((SCREEN_WIDTH - size[0]) // 2, (SCREEN_HEIGHT - size[1]) // 2)
            if game.state == gameState.DRAW:
                draw_popup(screen, size, pos, "Draw")
            else:
                draw_popup(screen, size, pos, f"{game.winner.name} wins")

def draw_X(screen, pos, step):
    center = ((step * pos[0]) + step / 2, (step * pos[1]) + step / 2)
    
    pygame.draw.line(screen, WHITE, (center[0] - 100, center[1] - 100), (center[0] + 100, center[1] + 100), 8)
    pygame.draw.line(screen, WHITE, (center[0] - 100, center[1] + 100), (center[0] + 100, center[1] - 100), 8)

def draw_O(screen, pos, step):
    center = ((step * pos[0]) + step / 2, (step * pos[1]) + step / 2)
    
    pygame.draw.circle(screen, WHITE, center, 100, 10)

def draw_popup(screen, size, pos, message):
    # Create a surface for the pop-up
    font = pygame.font.Font(None, 90)

    popup_surface = pygame.Surface(size)
    popup_surface.fill(GREY)

    # Render text for the pop-up
    text_surface = font.render(message, True, WHITE)
    text_rect = text_surface.get_rect(center=(size[0] // 2, size[1] // 2 - 20))
    popup_surface.blit(text_surface, text_rect)

    # Draw the pop-up to the screen
    screen.blit(popup_surface, pos)