from abc import ABC, abstractmethod
import numpy as np

class Move(ABC):
    name: str = ""
    player: str
    
    def __init__(self, name: str, player: str) -> None:
        super().__init__()
        self.name = name
        self.player = player
        
    def __eq__(self, __value: object) -> bool:
        return super().__eq__(__value)
        
          

class Board(ABC):
    state: np.ndarray
    legal_moves: list[Move]
    
    def __init__(self, board_size: tuple) -> None:
        super().__init__()
        self.state = np.ndarray(board_size)
        
    def is_legal_move(self, move: Move):
        return move in self.legal_moves
    
    @abstractmethod
    def make_move(self, move: Move):
        pass
    
    @abstractmethod
    def unmake_move(self, move: Move):
        pass
    
    
    
    