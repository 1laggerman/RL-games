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
    reward: float
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
        # function to be implemented by a child class, to create a move from input of the specific child class
        # for example: for a tictactoe_board, this function should read a user input and return a tictactoe_Move object
        pass        
        
    def move(self, move: Move):
        self.history.append(move)
        
        self.make_move(move)
        self.update_state(move)
        
        self.next_player()
    
    @abstractmethod
    def make_move(self, move: Move):
        # this function represents a move being done by a player.
        # it should update self.board, update self.legal_moves and update self.reward
        pass
    
    def unmove(self, move: Move = None):
        if move is None:
            move = self.history.pop()
            
        self.unmake_move(move)
        self.prev_player()
        
        self.state = gameState.ONGOING
        self.winner = ""
        
    @abstractmethod
    def unmake_move(self, move: Move):
        # this is a reverse function for self.make_move()
        # it should update self.board and update self.legal_moves
        pass
    
    @abstractmethod
    def update_state(self, last_move: Move):
        # this function checks if the game is over
        # it should ensure self.state and self.winner are correct for the state of the board
        pass
    
    # encodes the board to a numpy array. mainly useful for neural network models
    def encode(self) -> np.ndarray:
        board = deepcopy(self.board)
        player_states = np.array([board == player for player in self.players])
        empty_state = self.board == ' '
        enc: np.ndarray = np.concatenate([player_states, empty_state.reshape((1, *empty_state.shape))])
        return enc.astype(np.float32)
    
    def win(self):
        self.state = gameState.ENDED
        self.winner = self.curr_player
        self.reward = 1
        
    def draw(self):
        self.state = gameState.DRAW
        self.reward = 0
        
    def lose(self):
        self.state = gameState.ENDED
        self.winner = self.players[self.curr_player_idx - 1]
        self.reward = -1
    
    def __str__(self):
        return self.board
    
    def __repr__(self) -> str:
        return str(self)
    
    def map_move(self, move: Move) -> int:
        return int(move.name)
    
    
    
    