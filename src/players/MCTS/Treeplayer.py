from src.base import Action, Game, gameState, Player

from typing import Callable
from dataclasses import dataclass
from abc import ABC, abstractmethod
import tqdm
import time


@dataclass(init=True, frozen=False)
class SArgs:
    max_iters: int
    max_time: float
    max_depth: int
    C: float

    def __init__(self, max_iters: int = 100, max_time: float = float('inf'), max_depth: int = -1, C: float = 2) -> None:
        self.max_iters = max_iters
        self.max_time = max_time
        self.max_depth = max_depth
        self.C = C
        

class Node(ABC):
    """
    A genreral Node representing a game state in the MCTS search tree
    
    Attributes:
        * visits (int): The number of times this node has been explored(including child nodes)
        * eval (float): The evaluation of the node including child nodes
        * parent (Node): The parent node
        * children (list[tuple[Move, Node]]): The children of this node
        * untried_actions (list[Move]): The moves that have not been explored from this node
        * is_leaf (bool): Whether this node represents a game that has ended
        
    Methods provided by the abstract class:
        * backpropagate(ev: float) -> None: Backpropagate the evaluation to the parent node
        
    Methods to be implemented by a child class:
        * select_child(): Select the child node to explore
        * expand(game: Game, move: Move = None): Expand the tree by creating a new child node
        * evaluate(game: Game): Evaluate the node
        * update_rule(eval: float) -> None: updates the eval after receiving a n eval from a descendant Node
    """
    visits: int = 0
    eval: float = 0
    parent: 'Node'
    player: 'Player'
    children: list[tuple[Action, "Node"]]
    untried_actions: list[Action]
    is_terminal: bool = False
    
    def __init__(self, game: Game, parent: "Node" = None) -> None:
        self.untried_actions = game.legal_moves.copy()
        self.visits = 0
        self.eval = 0
        self.parent = parent
        self.children = list()
        self.player = game.curr_role
        self.is_terminal = False
        
        if game is not None:
            if game.state != gameState.ONGOING:
                self.is_terminal = True
            self.update_rule(self.evaluate(game))
            self.visits = 1

    @abstractmethod        
    def select_child(self) -> tuple[Action, "Node"]:
        """
        function to be implemented by a child class
        selects the child node to explore/exploit while searching the tree for the best move
        """
        pass
    
    @abstractmethod
    def evaluate(self, game: Game) -> float:
        """
        function to be implemented by a child class
        evaluates the current node, with respect to player
        
        Args:
            * game (Game): The current game state
            
        returns:
            * float: The evaluation of the node
        """
        pass
            
    # @abstractmethod
    # def expand(self, game: Game, move: Move = None) -> tuple[Move, "Node"]:
    #     """
    #     function to be implemented by a child class
    #     expands the tree by adding child node(or nodes) to this node
        
    #     Args:
    #         * game (Game): The current game state
    #         * move (Move, optional): The move to be explored. Defaults to None.
    #     """
    #     pass

    @abstractmethod
    def update_rule(self, new_eval: float):
        pass
           
    def backpropagate(self, eval: float, stop_at: 'Node' = None):
        """
        backpropagates the evaluation to all ancestors of this node
        calls method update_rule to update the rule
        
        Args:
            * eval (float): The evaluation to backpropagate.\n
        override to get a different behaviour if needed
        """
        self.visits += 1
        self.update_rule(eval)

        if self.parent is not None and self != stop_at:
            self.parent.backpropagate(eval)
            
    def __str__(self) -> str:
        return str(self.eval)
    
    def __repr__(self) -> str:
        return str(self)
            
class TreePlayer(Player, ABC):
    """
    A general MCTS player
    
    Attributes:
        * start_node (Node): The absolute root of the tree, from which all other nodes are descendants
        * root (Node): The current root of the tree, set to the current game state node
        * game (Game): reference to the Game object
        * search_iters (int): The max number of iterations to search the tree for (set to 1000 by default)
        * search_time (float): The max time to search the tree for (set to infinity by default)
        * max_depth (int): The max depth to search the tree for (set to -1 for unlimited depth)
        
    Methods provided by the abstract class:
        * get_move(): perfoems a search of the tree and returns the best move found using best()
        * calc_best_move(): builds a search tree by searching the tree for search_iters iterations or search_time seconds with at most max_depth depth
        * move(move: Move): updates self.root after a move was made in the game. creates a new Node and links to the tree if it doesnt exist yet
        
    Methods to be implemented by a child class:
        * create_node(untried_actions: list[Move], player: player): creates a new node for the tree
        * best(node: Node): returns the best move found in the tree
    """
    start_node: Node
    root: Node
    game: Game
    search_args: SArgs
    
    def __init__(self, game: Game, name: str, search_args: SArgs) -> None:
        super(TreePlayer, self).__init__(game, name)
        self.search_args = search_args
        self.start_node = None
        self.root = None
        self.game = game
        
    def get_move(self):
        """
        searches the tree by calling calc_best_move and returns the best move
        """
        if self.root == None:
            if self.game is None:
                raise ValueError("Board not set")
            self.root = self.expand(self.game, None)
            if self.start_node == None:
                self.start_node = self.root
        
        self.search_tree()
        for move, node in self.root.children:
            print(move, ': %.3f / %d' % (node.eval, node.visits))
        return self.best()[0]


    def search_tree(self):
        """
        calculates the best move by searching the tree
        stops if the search time is reached or the search_iters is reached
        
        * adds a minimum of root.untried_actions number of nodes to the tree
        * adds a maximum of self.search_iters number of nodes to the tree
        * doesnt add nodes of depth greater than self.max_depth
        """
        iters = self.search_args.max_iters
        curr_max_depth = 0
        start_time = time.time()
        
        while len(self.root.untried_actions) > 0:
            game = self.game
            self.expand(game, self.root)
            # (move, node) = self.root.expand(game)
            # game.make_move(move)
            # ev = node.evaluate(game)
            # game.unmake_move()
            # node.backpropagate(ev)
            iters -= 1
        
        # with tqdm.tqdm(total=max_iter)
        
        for _ in tqdm.tqdm(range(iters)):
            t0 = time.time()
            node = self.root
            game = self.game
            depth = 0
            
            while len(node.untried_actions) == 0 and not node.is_terminal:
                (move, node) = node.select_child(self.search_args.C)
                game.make_move(move)
                depth += 1
            
            ev = 0
            if not node.is_terminal:
                if self.search_args.max_depth <= 1 or depth + 1 < self.search_args.max_depth:
                    node = self.expand(game, node)
                    if depth + 1 > curr_max_depth:
                        curr_max_depth = depth + 1


            ev = node.evaluate(game)
            node.backpropagate(ev, stop_at=self.root)
                    
            
            for d in range(depth):
                self.game.unmake_move()
            
            # node.backpropagate(ev, stop_at=self.root)
            
            flag_time = time.time()
            if flag_time - start_time > self.search_args.max_time:
                break
    
    def update_state(self, move: Action):
        """
        updates the root node after a move is made
        if the child node doesnt exists, it creates it
        """
        found = False
        for m, node in self.root.children:
            if m == move:
                found = True
                self.root = node
                break
        
        if not found:
            (move, node) = self.expand(self.game, node, move)
            self.root = node
    
    @abstractmethod       
    def best(self) -> tuple[Action, Node]:
        """
        function to be implemented by a child class
        return the best move according to the current evaluation
        
        Returns:
            tuple[Move, Node]: The best move and the node that represents the game state after playing the move
        """
        pass

    @abstractmethod
    def expand(self, state: Game, parent: Node | None = None, move: Action | None = None) -> Node:
        # TODO: refactor the change to use Board instead
        """
        function to be implemented by a child class
        creates a new node with the given parameters (to make easier implementing a child class)
        
        Args:
            * untried_actions (list[Move]): The untried actions for the new node (the legal moves for the current player)
            * player (player): The player for the new node
            * parent (Node, optional): The parent node for the new node. Defaults to None.
        """
        pass