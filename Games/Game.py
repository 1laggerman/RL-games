from abc import ABC, abstractmethod
import numpy as np
import enum as Enum
from enum import Enum
import copy

class gameState(Enum):
    ENDED = 'E'
    DRAW = 'D'
    ONGOING = 'P'

class Move(ABC):
    name: str = ""
    
    def __init__(self, name: str) -> None:
        super(Move, self).__init__()
        self.name = name
        
    def __eq__(self, __value: "Move") -> bool:
        return self.name == __value.name
    
    def __str__(self) -> str:
        return self.name
    
    def __repr__(self):
        return str(self)
          

class Board(ABC):
    board: np.ndarray
    legal_moves: list[Move]
    players: list[str] = ['X', 'O']
    state: gameState = gameState.ONGOING
    winner: str = ""
    curr_player_idx: int = 0
    curr_player: str = players[0]
    history: list[Move] = []
    
    def __init__(self, board_size: tuple, players: list[str] = ['X', 'O']) -> None:
        super().__init__()
        self.board = np.full(board_size, fill_value=" ", dtype=str)
        self.players = copy.deepcopy(players)
        
    def is_legal_move(self, move: Move):
        return move in self.legal_moves
    
    def next_player(self):
        self.curr_player_idx = self.curr_player_idx + 1
        if self.curr_player_idx == self.players.__len__():
            self.curr_player_idx = 0
        self.curr_player = self.players[self.curr_player_idx]
        
    def prev_player(self):
        self.curr_player_idx = self.curr_player_idx - 1
        if self.curr_player_idx < 0:
            self.curr_player_idx = self.players.__len__() - 1
        self.curr_player = self.players[self.curr_player_idx]
        
    @abstractmethod
    def create_move(self, input: str) -> Move:
        pass        
        
    @abstractmethod
    def make_move(self, move: Move):
        pass
    
    @abstractmethod
    def unmake_move(self, move: Move = None):
        pass
    
    @abstractmethod
    def update_state(move: Move):
        pass
    
    def encode(self) -> np.ndarray:
        pass
    
    def __str__(self):
        return self.board
    
    def __repr__(self) -> str:
        return str(self)
    
    
    
    