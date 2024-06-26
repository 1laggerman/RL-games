from src.base import Move, Board, gameState, player

import random
import math
from copy import deepcopy
from abc import ABC, abstractmethod
import tqdm
import time

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
        * backpropagate(ev: float): Backpropagate the evaluation to the parent node
        
    Methods to be implemented by a child class:
        * select_child(): Select the child node to explore
        * expand(board: Board, move: Move = None): Expand the tree by creating a new child node
        * evaluate(board: Board): Evaluate the node
    """
    visits: int = 0
    eval: float = 0
    parent: 'Node'
    player: 'player'
    children: list[tuple[Move, "Node"]]
    untried_actions: list[Move]
    is_leaf: bool = False
    
    def __init__(self, untried_actions: list[Move], player: 'player', parent: "Node" = None) -> None:
        self.visits = 0
        self.eval = 0
        self.parent = parent
        self.children = list()
        self.player = player
        self.untried_actions = deepcopy(untried_actions)
        self.is_leaf = False
        if len(self.untried_actions) == 0:
            self.is_leaf = True

    @abstractmethod        
    def select_child(self) -> tuple[Move, "Node"]:
        """
        function to be implemented by a child class
        selects the child node to explore/exploit while searching the tree for the best move
        """
        pass
    
    @abstractmethod
    def evaluate(self, board: Board) -> float:
        """
        function to be implemented by a child class
        evaluates the current node, with respect to player
        
        Args:
            * board (Board): The current game state
            
        returns:
            * float: The evaluation of the node
        """
        pass
            
    @abstractmethod
    def expand(self, board: Board, move: Move = None) -> tuple[Move, "Node"]:
        """
        function to be implemented by a child class
        expands the tree by adding child node(or nodes) to this node
        
        Args:
            * board (Board): The current game state
            * move (Move, optional): The move to be explored. Defaults to None.
        """
        pass
        
    def backpropagate(self, eval: float):
        """
        backpropagates the evaluation to all ancestors of this node
        
        Args:
            * eval (float): The evaluation of the node, expected to be in the range [-1, 1].\n
            override to get a different behaviour if needed
        """
        self.visits += 1
        
        self.eval += eval
        
        if self.parent:
            self.parent.backpropagate(-eval)
            
    def __str__(self) -> str:
        return str(self.eval)
    
    def __repr__(self) -> str:
        return str(self)
            
class TreePlayer(player, ABC):
    """
    A general MCTS player
    
    Attributes:
        * start_node (Node): The absolute root of the tree, from which all other nodes are descendants
        * root (Node): The current root of the tree, set to the current game state node
        * board (Board): reference to the game board
        * search_iters (int): The max number of iterations to search the tree for (set to 1000 by default)
        * search_time (float): The max time to search the tree for (set to infinity by default)
        * max_depth (int): The max depth to search the tree for (set to -1 for unlimited depth)
        
    Methods provided by the abstract class:
        * get_move(): perfoems a search of the tree and returns the best move found using best()
        * calc_best_move(): builds a search tree by searching the tree for search_iters iterations or search_time seconds with at most max_depth depth
        * move(move: Move): updates self.root after a move was made in the board. creates a new Node and links to the tree if it doesnt exist yet
        
    Methods to be implemented by a child class:
        * create_node(untried_actions: list[Move], player: player): creates a new node for the tree
        * best(node: Node): returns the best move found in the tree
    """
    start_node: Node
    root: Node
    board: Board
    search_iters: int
    search_time: float
    max_depth: int
    
    def __init__(self, game_board: Board, name: str, search_iters: int = 1000, search_time: float = float('inf'), max_depth: int = -1) -> None:
        super(TreePlayer, self).__init__(game_board, name)
        # self.root = None
        self.start_node = self.create_node(game_board.legal_moves, player=game_board.curr_player)
        self.root = self.start_node
        
    def get_move(self):
        """
        searches the tree by calling calc_best_move and returns the best move
        """
        if self.root == None:
            self.root = self.create_node(self.board.legal_moves, player=self.board.curr_player, parent=None)
        
        self.calc_best_move()
        return self.best()[0]
        
    def calc_best_move(self):
        """
        calculates the best move by searching the tree
        stops if the search time is reached or the search_iters is reached
        
        * adds a minimum of root.untried_actions number of nodes to the tree
        * adds a maximum of self.search_iters number of nodes to the tree
        * doesnt add nodes of depth greater than self.max_depth
        """
        max_d = 0
        start_time = time.time()
        
        while len(self.root.untried_actions) > 0:
            board = deepcopy(self.board)
            (move, node) = self.root.expand(board)
            board.make_move(move)
            ev = node.evaluate(board)
            # self.board.unmake_move()
            node.backpropagate(ev)
            self.search_iters -= 1
        
        # with tqdm.tqdm(total=max_iter)
        
        for _ in tqdm.tqdm(range(self.search_iters)):
            t0 = time.time()
            node = self.root
            board = deepcopy(self.board)
            depth = 0
            
            t1 = time.time()
            while len(node.untried_actions) == 0 and not node.is_leaf:
                (move, node) = node.select_child()
                board.make_move(move)
                depth += 1
            
            t2 = time.time()
            
            ev = 0
            if not node.is_leaf:
                if self.max_depth <= 1 or depth + 1 < self.max_depth:
                    (move, node) = node.expand(board)
                    board.make_move(move)
                    depth += 1
                    
            t3 = time.time()
            ev = node.evaluate(board)
            
            t4 = time.time()
            # for d in range(depth):
            #     self.board.unmake_move()
            
            node.backpropagate(ev)
            if depth > max_d:
                max_d = depth
                
            t5 = time.time()
            
            # print("deepcopy time: ", t1 - t0)
            # print("find node time: ", t2 - t1)
            # print("expand node time: ", t3 - t2)
            # print("eval node time: ", t4 - t3)
            # print("backpropagate node time: ", t5 - t4)
            
            flag_time = time.time()
            
            if flag_time - start_time > self.search_time:
                break
    
    def move(self, move: Move):
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
            (move, node) = self.root.expand(self.board, move)
            self.root = node
        # self.board.make_move(move)
        return
    
    @abstractmethod       
    def best(self) -> tuple[Move, Node]:
        """
        function to be implemented by a child class
        return the best move according to the current evaluation
        
        Returns:
            tuple[Move, Node]: The best move and the node that represents the game state after playing the move
        """
        pass

    @abstractmethod
    def create_node(self, untried_actions: list[Move], player: player, parent: Node = None) -> Node:
        """
        function to be implemented by a child class
        creates a new node with the given parameters (to make easier implementing a child class)
        
        Args:
            * untried_actions (list[Move]): The untried actions for the new node (the legal moves for the current player)
            * player (player): The player for the new node
            * parent (Node, optional): The parent node for the new node. Defaults to None.
        """
        pass