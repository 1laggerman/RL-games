from abc import ABC, abstractmethod
from Games.Game import Move, Board, gameState

class player(ABC):
    
    def __init__(self, game_board: Board) -> None:
        self.board = game_board

    @abstractmethod
    def move(self, move: Move):
        pass