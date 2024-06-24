from abc import ABC, abstractmethod
import numpy as np
import enum as Enum
from enum import Enum
from copy import deepcopy, copy


class player(ABC):
    # TODO: add descrition
    board: 'Board'
    name: str
    
    def __init__(self, game_board: 'Board', name: str) -> None:
        self.board = game_board
        self.name = name

    @abstractmethod
    def get_move(self):
        # TODO: add descrition
        pass
    
    def move(self, move: 'Move'):
        # TODO: add descrition
        pass
    
def bind(board: 'Board', players: list['player']):
    # TODO: add descrition
    if len(players) == 0:
        print("No players")
        return
    
    board.players = players
    board.curr_player = players[board.curr_player_idx]
    
    for player in players:
        player.board = board

def play(board: 'Board', players: list['player']):
    # TODO: add descrition
    bind(board, players)
    while board.state == gameState.ONGOING:
        move = board.curr_player.get_move()
        if move is None:
            print("Invalid move")
            return
        board.make_move(move)
        
    if board.state == gameState.ENDED:
        print(f"Winner: {board.winner.name}")
        print(f"Reward: {board.reward}")
    else:
        print("Draw")
        
class gameState(Enum):
    ENDED = 'E'
    DRAW = 'D'
    ONGOING = 'P'

class Move(ABC):
    # TODO: add descrition
    name: str = ""
    
    def __init__(self, name: str) -> None:
        super(Move, self).__init__()
        self.name = name.replace(" ", "")
        # self.name = name.join("")
        
    def __eq__(self, __value: "Move") -> bool:
        return self.name == __value.name
    
    def __str__(self) -> str:
        return '(' + self.name + ')'
    
    def __repr__(self):
        return str(self)

class Board(ABC):
    # TODO: add descrition
    board: np.ndarray
    legal_moves: list[Move]
    players: list['player']
    state: gameState
    reward: float
    winner: 'player'
    curr_player_idx: int
    curr_player: 'player'
    history: list[Move]
    
    def __init__(self, board_size: tuple, players: list['player'] = []) -> None:
        super().__init__()
        self.state = gameState.ONGOING
        self.history = []
        self.board = np.full(board_size, fill_value=" ", dtype=str)
        self.reward = 0
        
        # if len(players) == 0:
        #     players = [(self, "O")]
        self.players = players
        
        for p in self.players:
            p.board = self
            
        self.winner = None
        self.curr_player_idx = 0
        if len(players) > 0:
            self.curr_player = players[0]
        
    def is_legal_move(self, move: Move):
        # TODO: add descrition
        return move in self.legal_moves
    
    def next_player(self):
        # TODO: add descrition
        self.curr_player_idx = self.curr_player_idx + 1
        if self.curr_player_idx == self.players.__len__():
            self.curr_player_idx = 0
        self.curr_player = self.players[self.curr_player_idx]
        
    def prev_player(self):
        # TODO: add descrition
        self.curr_player_idx = self.curr_player_idx - 1
        if self.curr_player_idx < 0:
            self.curr_player_idx = self.players.__len__() - 1
        self.curr_player = self.players[self.curr_player_idx]
        
    def make_move(self, move: Move):
        # TODO: add descrition
        self.history.append(move)
        self.update_state(move)
        self.next_player()
    
    def unmake_move(self, move: Move = None):
        # TODO: add descrition
        move = self.history.pop()
        self.reverse_state(move)
        self.prev_player()
        
    @abstractmethod
    def create_move(self, input: str) -> Move:
        """
        function to be implemented by a child class, to create a move from input of the specific child class

        Args:
            input (str): A string encoding of the move

        Returns:
            Move: a move object of the specific child class
        """
        pass        
    
    @abstractmethod
    def update_state(self, move: Move) -> float:
        """
        this function represents a move being done by a player.
        
        Args:
            move (Move): the move being made
        
        Effect:
            self.board - update board matrix visuals\n
            self.legal_moves - update legal moves to be made\n
            self.reward - update the total reward from this game\n
            self.state - is the game over?\n
            self.winner - if the game is over - update winner\n
            
        Returns:
            the reward for the move
        """
        pass
    
    @abstractmethod
    def reverse_state(self, move: Move):
        # TODO: add descrition
        # this function represents a move being undone by a player.
        # it should update self.board, self.legal_moves, self.reward, self.state, self.winner
        pass
    
    # encodes the board to a numpy array. mainly useful for neural network models
    def encode(self) -> np.ndarray:
        # TODO: add descrition
        board = deepcopy(self.board)
        player_states = np.array([board == player for player in self.players])
        empty_state = self.board == ' '
        enc: np.ndarray = np.concatenate([player_states, empty_state.reshape((1, *empty_state.shape))])
        return enc.astype(np.float32)
    
    def win(self):
        # TODO: add descrition
        self.state = gameState.ENDED
        self.winner = self.curr_player
        self.reward = 1
        
    def draw(self):
        # TODO: add descrition
        self.state = gameState.DRAW
        self.reward = 0
        
    def lose(self):
        # TODO: add descrition
        self.state = gameState.ENDED
        self.winner = self.players[self.curr_player_idx - 1]
        self.reward = -1
    
    def __str__(self):
        return str(self.board)
    
    def __repr__(self) -> str:
        return str(self)
    
    def map_move(self, move: Move) -> int:
        # TODO: add descrition
        return int(move.name)
    
    
    def __deepcopy__(self, memo):
        
        cls = self.__class__
        result = cls.__new__(cls)
        
        memo[id(self)] = result
        
        result.board = deepcopy(self.board, memo)
        result.legal_moves = deepcopy(self.legal_moves)
        result.history = deepcopy(self.history)
        result.state = self.state
        result.reward = self.reward
        result.curr_player_idx = self.curr_player_idx
        result.curr_player = self.curr_player
        result.players = self.players
        result.winner = self.winner
        
        return result