import sys
sys.path.insert(0, '') 
from abc import ABC, abstractmethod
import package.Games.Game as game

class player(ABC):
    board: game.Board
    name: str
    
    def __init__(self, game_board: game.Board, name: str) -> None:
        self.board = game_board
        self.name = name

    @abstractmethod
    def get_move(self):
        pass