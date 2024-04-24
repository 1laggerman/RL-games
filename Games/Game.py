from abc import ABC, abstractmethod
import numpy as np
import enum as Enum
from enum import Enum
from copy import deepcopy

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
    state: gameState
    winner: str
    curr_player_idx: int
    curr_player: str
    history: list[Move]
    
    def __init__(self, board_size: tuple, players: list[str] = ['X', 'O']) -> None:
        super().__init__()
        self.board = np.full(board_size, fill_value=" ", dtype=str)
        self.players = deepcopy(players)
        self.state = gameState.ONGOING
        self.winner = ""
        self.curr_player_idx = 0
        self.curr_player = players[0]
        self.history = []
        
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
        board = deepcopy(self.board)
        player_states = np.array([board == player for player in self.players])
        empty_state = self.board == ' '
        enc: np.ndarray = np.concatenate([player_states, empty_state.reshape((1, *empty_state.shape))])
        return enc.astype(np.float32)
    
    def __str__(self):
        return self.board
    
    def __repr__(self) -> str:
        return str(self)
    
    # def __copy__(self):
    #     cls = self.__class__
    #     result = cls.__new__(cls)
    #     result.__dict__.update(self.__dict__)
    #     return result

    # def __deepcopy__(self, memo):
    #     cls = self.__class__
    #     result = cls.__new__(cls)
    #     memo[id(self)] = result
    #     for k, v in self.__dict__.items():
    #         setattr(result, k, deepcopy(v, memo))
    #     return result
    
    
    
    