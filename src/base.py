from abc import ABC, abstractmethod
import numpy as np
import enum as Enum
from enum import Enum
from copy import deepcopy, copy
from pathlib import Path
import torch

current_file_path = Path(__file__).resolve()


# TODO: add pieace wrapper class to hold pieaces of the same type for more effeicient encoding
class Player(ABC):
    """
    A basic abstract player class
    
    Args:
        game_board (Board): the board the player is playing on
        name (str): the name of the player
        pieces (list[piece]): the pieces the player has (init to [])

    Abstract methods to implement:
        get_move(self) -> Move
        move(self, move: Move) -> None
    """
    board: 'Game'
    name: str
    reward: float
    pieces: list['Piece']
    
    def __init__(self, game_board: 'Game', name: str) -> None:
        self.board = game_board
        self.name = name
        self.reward = 0
        self.pieces = []

    def recv_reward(self, reward: float):
        """
        updates the reward of the player for the game
        """
        self.reward += reward

    def inform_player(self, move: 'Move'):
        """
        wrapper function for update_state that receives the reward and updates the state

        Args:
            * move (Move): the move being played
        """
        if move.reward is not None:
            self.recv_reward(move.reward)
        self.update_state(move)

    def encode(self) -> np.ndarray:
        """
        returns the encoding of the player pieaces

        Returns:
            * np.ndarray: the encoding of the player
        """
        return np.array([self.board == piece for piece in self.pieces])


    @abstractmethod
    def get_move(self) -> 'Move':
        """
        chooses a move to be played using self.board

        Returns:
            * Move: the move to be played
        """
        pass
    
    @abstractmethod
    def update_state(self, move: 'Move'):
        """
        inner function to update player's state after a move is played
        
        Args:
            * move (Move): the move being played
        """
        pass
        
class gameState(Enum):
    """
    Enum to represent the state of the game
    
    Attributes:
    -----------
        ENDED: the game has ended
        DRAW: the game ended in a draw
        ONGOING: the game is still ongoing
    """
    ENDED = 'E'
    DRAW = 'D'
    ONGOING = 'P'
    
class Piece(ABC):
    """
    Abstract pieace class
        
    Attributes:
        * name (str): string representation of the piece
        * player (player): player that owns the piece
        * location (tuple[int]): location of the piece on the board
    """
    name: str
    player: 'Player'
    location: tuple[int]
    
    def __init__(self, name: str, player: 'Player', location: tuple[int]) -> None:
        self.player = player
        self.name = name
        self.location = location
        
    def __eq__(self, other) -> bool:
        if isinstance(other, Piece):
            return self.name == other.name
        elif isinstance(other, Player):
            return self.player.name == other.name
        elif isinstance(other, str):
            return self.name == other
            
        elif type(other) is str:
            return self.name == other
        
    def __str__(self) -> str:
        return self.name
    
    def __repr__(self) -> str:
        return str(self)
    
    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        result.name = self.name
        result.player = self.player
        result.location = copy(self.location)
        return result

class Move(ABC):
    """
    Abstract move class
    
    Attributes:
    -----------
        * name (str): String representation of the move
        * reward (float): optional reward for the move(if known). 0 by default.
        * player (player): optional player that made the move. None by default. (usualy not needed as board holds curr_player)
        * src_location: tuple[int]: the square where the piece is moved from
        * dest_location: tuple[int]: the square where the piece is moved to
    """
    name: str
    src_location: tuple[int]
    dest_location: tuple[int]
    reward: float = 0
    piece: Piece
    
    def __init__(self, name: str, player: 'Player' = None, reward: float = 0) -> None:
        super(Move, self).__init__()
        self.name = name.replace(" ", "") # clean move name
        self.reward = reward
        
    def __eq__(self, __value: "Move") -> bool:
        return self.name == __value.name
    
    def __str__(self) -> str:
        return '(' + self.name + ')'
    
    def __repr__(self):
        return str(self)

class Game(ABC):
    """
    Abstract Game class
    
    Attributes:
    -----------
        * board (np.ndarray[Piece]): holds visualization of the board - init to None pieces\n
        * legal_moves (list[Move]): the legal moves for the current player with respect to the board - requires initialization\n
        * players (list[player]): the players playing the game - provided by the user\n
        * state (gameState): the current state of the game - init to ONGOING\n
        * reward (float): the total reward of the game so far - init to 0\n
        * winner (player): the winner of the game - init to None\n
        * curr_player_idx (int): the index of the current player in self.players - init to 0\n
        * curr_player (player): the current player - init to players[0]\n
        * history (list[Move]): the history of moves played - init to []\n
        
    Methods provided:
    -----------------
    
        * is_legal_move(self, move: Move) -> bool:\n
            checks if the move is in legal_moves\n
            you may want to override __eq__ for Move for more accurate performance\n
    
        * alert_players(self, move: Move) -> None\n
            informs the players of the move\n
            
        * make_move(self, move: Move) -> None:\n
            adds the move to history, calls update_state, and updates curr_player\n
            
        * unmake_move(self) -> None:\n
            removes the last move from history, calls reverse_state, and updates curr_player\n   
        
        * next_player(self) -> None:\n
            updates curr_player_idx(+1 mod len(players)) and curr_player\n
            
        * prev_player(self) -> None:\n
            updates curr_player_idx(+1 mod len(players)) and curr_player\n

        * encode(self) -> np.ndarray:\n
            encodes the board into a numpy array for ML usage.\n
            defualt encodes by layer for each pieace and a legal_moves layer\n
            
        * win(self, player: player) -> None:\n
            updates the state of the game to ENDED and sets the winner to player and reward 1\n
            
        * lose(self, player: player) -> None:\n
            updates the state of the game to ENDED and sets the winner to other player and reward -1\n
            
        * draw(self) -> None:\n
            updates the state of the game to ENDED and sets the winner to None and reward 0\n
            
    Method to override:
    -------------------
    
        * create_move(self, move: Move) -> None:\n
            makes a move on the board and updates the state of the game accordingly\n
            
        * update_state(self, move: Move) -> None:\n
            updates the state of the game based on the current board\n
            
        * reverse_state(self, move: Move) -> None:\n
            reverses the given move\n
    """
    board: np.ndarray[Piece]
    legal_moves: list[Move]
    all_moves: list[Move]
    players: list['Player']
    state: gameState
    reward: float
    winner: 'Player'
    curr_player_idx: int
    curr_player: 'Player'
    history: list[Move]
    
    def __init__(self, board_size: tuple, players: list['Player'] = []) -> None:
        super().__init__()
        self.state = gameState.ONGOING
        self.history = []
        self.board = np.full(board_size, fill_value=None, dtype=object)
        self.reward = 0
    
        self.players = players
        
        for p in self.players:
            p.board = self
            
        self.winner = None
        self.curr_player_idx = 0
        if len(players) > 0:
            self.curr_player = players[0]
            
    @abstractmethod
    def create_move(self, input: str) -> Move:
        """
        function to be implemented by a child class
        creates a move from input of the specific child class

        Args:
        -----
            * input (str): A string encoding of the move

        Returns:
        --------
            Move: a move object of the specific child class
        """
        pass        
    
    @abstractmethod
    def update_state(self, move: Move) -> float:
        """
        function to be implemented by a child class
        updates self object after a move is made
        
        Args:
        -----
            * move (Move): the move being made
        
        Required Effects:
        -----------------
            * self.board - update board matrix visuals\n
            * self.legal_moves - update legal moves according to your game\n
            * self.reward - update the total reward from this game\n
            * self.state - is the game over?\n
            * self.winner - if the game is over - update winner\n
            
        Returns:
        --------
            the imidiate reward for the move
        """
        pass
    
    @abstractmethod
    def reverse_state(self, move: Move):
        """
        function to be implemented by a child class
        this function is the reverse of update_state.
        
        Args:
        -----
            * move (Move): the move that was made last
        
        Required Effects:
        -----------------
            * self.board - reverse board matrix visuals\n
            * self.legal_moves - reverse legal moves according to your game\n
            * self.reward - reverse the total reward from this game\n
        """
        pass
    
    def alert_players(self, move: Move):
        """
        informs all players of a move being made
        
        Args:
        -----
            * move (Move): the move being made
        """
        for player in self.players:
            player.update_state(move)    
        
    def is_legal_move(self, move: Move) -> bool:
        """
        checks if the move is in legal_moves

        Args:
        -----
            * move (Move): the move to search for

        Returns:
        --------
            bool: True if the move is in legal_moves, False otherwise
        """
        return move in self.legal_moves
    
    def next_player(self):
        """
        calculates the next player by incrementing the current player index and updating the current player
        """
        self.curr_player_idx = (self.curr_player_idx + 1) % len(self.players)
        self.curr_player = self.players[self.curr_player_idx]
        
    def prev_player(self):
        """
        calculates the previous player by decrementing the current player index and updating the current player
        """
        self.curr_player_idx = (self.curr_player_idx - 1) % len(self.players)
        self.curr_player = self.players[self.curr_player_idx]
        
    def make_move(self, move: Move):
        """
        shell function to make a move, adds the move to history, calls update_state, and updates curr_player
        """
        self.history.append(move)
        self.update_state(move)
        self.next_player()
    
    def unmake_move(self, move: Move = None):
        """
        shell function to unmake a move, removes the last move from history, calls reverse_state, and updates curr_player
        """
        move = self.history.pop()
        self.reverse_state(move)
        self.prev_player()
    
    def encode(self, perspective: Player = None) -> np.ndarray:
        """
        encodes the board to a numpy array for use of neural network models
        
        defualt encoding(None perspective):
        -----------------
            - layer for empty space
            - layer for each pieace, ordered by player\n
        defualt encoding(with perspective):
        -----------------
            - layer for perspective player pieaces
            - layer for empty space
            - layer for each pieace, ordered by player\n
        """
        player_states = np.array([player.encode() for player in self.players if player != perspective])
        empty_state = self.board == None
        layers = [player_states, empty_state.reshape((1, *empty_state.shape))]
        if perspective is not None:
            layers.append(perspective.encode())
        enc: np.ndarray = np.concatenate(layers)
        return enc.astype(np.float32)
        # TODO: encode curr player
    
    def win(self):
        """
        turns the game to a win for the current player
        """
        self.state = gameState.ENDED
        self.winner = self.curr_player
        self.reward = 1
        for player in self.players:
            if player == self.winner:
                player.recv_reward(self.reward)
            else:
                player.recv_reward(-self.reward)
        
    def draw(self):
        """
        turns the game to a draw
        """
        self.state = gameState.DRAW
        self.reward = 0
        for player in self.players:
            player.recv_reward(self.reward)
        
    def lose(self):
        """
        turns the game to a loss for the current player
        """
        self.state = gameState.ENDED
        self.winner = self.players[self.curr_player_idx - 1]
        self.reward = -1
        for player in self.players:
            if player == self.winner:
                player.recv_reward(self.reward)
            else:
                player.recv_reward(-self.reward)

    def map_move(self, move: Move) -> int:
        """
        hash function for moves

        Args:
        -----
            * move (Move): the move to be hashed

        Returns:
        --------
            mapping of each move according to the unravled index of the destination square of the move
        """
        return np.ravel_multi_index(move.dest_location, self.board.shape)
    
    def __str__(self):
        return str(self.board)
    
    def __repr__(self) -> str:
        return str(self)
    
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
    
def bind(game: 'Game', players: list['Player']):
    """
    binds game and players to each other
    couses each side to hold a reference to the other

    Args:
    -----
        * board (Board): the board to be played
        * players (list[player]): the players playing the game
    """
    if len(players) == 0:
        print("No players")
        return
    
    game.players = players
    game.curr_player = players[game.curr_player_idx]
    
    for player in players:
        player.board = game

def play(game: 'Game', players: list['Player']):
    """
    simulates a simple game loop between any 2 players

    Args:
    -----
        * game (Game): the game that is being played
        * players (list[player]): the players playing the game
    """
    bind(game, players)
    while game.state == gameState.ONGOING:
        move = game.curr_player.get_move()
        if move is None:
            print("Invalid move")
            return
        game.make_move(move)
        game.alert_players(move)
        
    if game.state == gameState.ENDED:
        print(f"Winner: {game.winner.name}")
        print(f"Reward: {game.reward}")
    else:
        print("Draw")

class Network(ABC):
    def __init__(self, game: 'Game') -> None:
        self.game = game
        enc = self.game.encode()
        self.in_features = enc.shape[0]

    @abstractmethod
    def evaluate(self) -> float:
        pass

    @abstractmethod
    def update(self) -> None:
        pass