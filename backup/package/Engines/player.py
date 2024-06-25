from package.Games.Game import Board, Move, gameState
from abc import ABC, abstractmethod

class player(ABC):
    board: Board
    name: str
    
    def __init__(self, game_board: Board, name: str) -> None:
        self.board = game_board
        self.name = name

    @abstractmethod
    def get_move(self):
        pass