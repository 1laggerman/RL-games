from abc import ABC, abstractmethod
import numpy as np
import enum as Enum
from enum import Enum
from copy import deepcopy, copy
from pathlib import Path
import torch

current_file_path = Path(__file__).resolve()

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

class Role():
    pieces: list['Piece']
    reward: float
    name: str
    player: 'Player'

    def __init__(self, name: str, pieces: list['Piece'] = [], initial_reward: float = 0, player: 'Player' = None) -> None:
        self.name = name
        self.reward = initial_reward
        self.pieces = pieces.copy()
        self.player = player

    def recv_reward(self, reward: float):
        """
        updates the reward of the player for the game
        """
        self.reward += reward

    def inform_player(self, action: 'Action'):
        """
        wrapper function for update_state that receives the reward and updates the state

        Args:
            * move (Move): the move being played
        """
        for role, reward in action.reward:
            if role == self:
                self.recv_reward(reward)
                
        if self.player is not None:
            self.player.update_state(action)

    def get_move(self) -> 'Action':
        """
        chooses a move to be played using self.board

        Returns:
            * Move: the move to be played
        """
        if self.player is None:
            return None
        return self.player.get_move()
    
    def __eq__(self, value: object) -> bool:
        if isinstance(value, Role):
            return self.name == value.name
        elif isinstance(value, str):
            return self.name == value
        return False
    
    def __str__(self) -> str:
        return self.name
    
    def __repr__(self) -> str:
        return str(self)

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
    name: str
    role: Role
    
    def __init__(self, game: 'Game', name: str) -> None:
        self.game = game
        self.name = name

    @abstractmethod
    def get_move(self) -> 'Action':
        """
        chooses a move to be played using self.board

        Returns:
            * Move: the move to be played
        """
        pass
    
    @abstractmethod
    def update_state(self, action: 'Action'):
        """
        inner function to update player's state after a move is played
        
        Args:
            * move (Move): the move being played
        """
        pass
    
class Piece(ABC):
    """
    Abstract pieace class
        
    Attributes:
        * name (str): string representation of the piece
        * player (player): player that owns the piece
        * location (tuple[int]): location of the piece on the board
    """
    name: str
    role: Role
    location: tuple[int]
    
    def __init__(self, name: str, role: Role, location: tuple[int] = None) -> None:
        self.role = role
        self.name = name
        self.location = location
        
    def __eq__(self, other) -> bool:
        if isinstance(other, Piece):
            return self.name == other.name
        elif isinstance(other, Role):
            return self.role.name == other.name
        elif isinstance(other, Player):
            return self.role.name == other.role.name
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
        result.role = self.role
        result.location = copy(self.location)
        return result

class Move:
    moved_piece: Piece
    src_location: tuple[int]
    dest_location: tuple[int]
    
    def __init__(self, moved_piece: Piece, to_location: tuple[int] | None) -> None:
        self.moved_piece = moved_piece
        self.src_location = moved_piece.location
        self.dest_location = to_location

class Action(ABC):
    name: str
    affects: list[Move]
    
    def __init__(self, name: str, affects: list[Move]) -> None:
        super(Action, self).__init__()
        self.name = name.replace(" ", "") # clean move name
        self.affects = affects
        
    def __eq__(self, __value: "Action") -> bool:
        return self.name == __value.name
    
    def __str__(self) -> str:
        return '(' + self.name + ')'
    
    def __repr__(self):
        return str(self)

class Game(ABC):
    board: np.ndarray[Piece]
    legal_actions: list[Action]
    all_actions: list[Action]
    roles: list[Role]
    state: gameState
    winner: 'Role'
    curr_role_idx: int
    curr_role: 'Role'
    history: list[tuple[Action, list[tuple[Role, float]]]]
    
    def __init__(self, board_size: tuple, roles: list[Role]) -> None:
        super().__init__()
        self.state = gameState.ONGOING
        self.history = []
        self.board = np.full(board_size, fill_value=None, dtype=object)
    
        self.roles = roles
            
        self.winner = None
        self.curr_role_idx = 0
        if len(roles) > 0:
            self.curr_role = roles[0]
            
    @abstractmethod
    def create_action(self, input: str) -> Action:
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
    def update_state(self, action: Action) -> list[tuple[Role, float]]:
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
            * self.reward - send rewards to players\n
            * self.state - is the game over?\n
            * self.winner - if the game is over - update winner\n
            
        Returns:
        --------
            the imidiate reward for the move
        """
        pass
    
    @abstractmethod
    def reverse_state(self, action: Action):
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
    
    def alert_players(self, action: Action):
        """
        informs all players of a move being made
        
        Args:
        -----
            * move (Move): the move being made
        """
        for role in self.roles:
            role.inform_player(action)  
        
    def is_legal_action(self, action: Action) -> bool:
        """
        checks if the move is in legal_moves

        Args:
        -----
            * move (Move): the move to search for

        Returns:
        --------
            bool: True if the move is in legal_moves, False otherwise
        """
        return action in self.legal_actions
    
    def next_player(self):
        """
        calculates the next player by incrementing the current player index and updating the current player
        """
        self.curr_role_idx = (self.curr_role_idx + 1) % len(self.roles)
        self.curr_role = self.roles[self.curr_role_idx]
        
    def prev_player(self):
        """
        calculates the previous player by decrementing the current player index and updating the current player
        """
        self.curr_role_idx = (self.curr_role_idx - 1) % len(self.roles)
        self.curr_role = self.roles[self.curr_role_idx]
        
    def make_action(self, action: Action):
        """
        shell function to make a move, adds the move to history, calls update_state, and updates curr_player
        """
        rewards = self.update_state(action)
        self.history.append((action, rewards))
        for role, reward in rewards:
            role.recv_reward(reward)
        self.next_player()
    
    def unmake_action(self):
        """
        shell function to unmake a move, removes the last move from history, calls reverse_state, and updates curr_player
        """
        action, rewards = self.history.pop()
        self.reverse_state(action)
        for role, reward in rewards:
            role.recv_reward(-reward)
        self.prev_player()
    
    def encode(self) -> np.ndarray:
        """
        encodes the board to a numpy array for use of neural network models
        
        defualt encoding:
        -----------------
            - layer for each pieace, ordered by player\n
            - layer for empty spaces\n
        """
        # board = deepcopy(self.board)
        player_states = np.array([self.board == role for role in self.roles])
        empty_state = self.board == None
        enc: np.ndarray = np.concatenate([player_states, empty_state.reshape((1, *empty_state.shape))])
        return enc.astype(np.float32)
        # TODO: encode curr player
    
    def win(self, role: Role | None = None):
        """
        turns the game to a win for the current player(defualt current player)
        """
        self.state = gameState.ENDED
        self.winner = self.curr_role

        rewards = []

        for role in self.roles:
            if role == self.winner:
                rewards.append((role, 1))
            else:
                rewards.append((role, -1))

        return rewards

    
    def draw(self):
        """
        turns the game to a draw
        """
        self.state = gameState.DRAW
        self.reward = 0
        for role in self.roles:
            role.recv_reward(self.reward)

        rewards = []

        return rewards

    def map_move(self, action: Action) -> int:
        """
        hash function for moves

        Args:
        -----
            * move (Move): the move to be hashed

        Returns:
        --------
            int: the index of the action inside self.all_actions
        """
        return self.all_actions.index(action)
    
    def __str__(self):
        return str(self.board)
    
    def __repr__(self) -> str:
        return str(self)
    
    def __deepcopy__(self, memo):
        
        cls = self.__class__
        result = cls.__new__(cls)
        
        memo[id(self)] = result
        
        result.board = deepcopy(self.board, memo)
        result.legal_actions = deepcopy(self.legal_actions)
        result.history = deepcopy(self.history)
        result.state = self.state
        result.reward = self.reward
        result.roles = self.roles
        result.curr_role = self.curr_role
        result.roles = self.roles
        result.winner = self.winner
        
        return result
    
def bind_all(game: 'Game', players: list['Player']):
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
    
    for i in range(len(game.roles)):
        game.roles[i].player = players[i]
        players[i].game = game
    
def bind(game: 'Game', player: 'Player', role: Role):
    """
    binds game and player to each other
    couses the player to hold a reference to the game)
    """
    player.game = game
    role.player = player


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
        move = game.curr_role.get_move()
        if move is None:
            print("Invalid move")
            return
        game.make_action(move)
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