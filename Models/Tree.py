from Games.Game import Move, Board, gameState
from enum import Enum
import random
import math
from copy import deepcopy
from abc import ABC, abstractmethod

class Node(ABC):
    visits: int = 0
    eval: float = 0
    parent: "Node"
    player: str
    children: list[tuple[Move, "Node"]]
    untried_actions: list[Move]
    is_leaf: bool = False
    
    def __init__(self, untried_actions: list[Move], player: str, parent: "Node" = None) -> None:
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
        pass
    
    @abstractmethod
    def evaluate(self, board: Board) -> float:
        pass
            
    @abstractmethod
    def expand(self, board: Board, move: Move = None) -> tuple[Move, "Node"]:
        pass
        
    def backpropagate(self, eval: float):
        self.visits += 1
        
        self.eval += eval
        
        if self.parent:
            self.parent.backpropagate(1-eval)
            
    def __str__(self) -> str:
        return str(self.eval)
    
    def __repr__(self) -> str:
        return str(self)
            
class SearchTree(ABC):
    root: Node
    board: Board
    
    def __init__(self, game_board: Board) -> None:
        self.board = game_board
        # self.root = Node(copy.deepcopy(game_board.legal_moves), player=game_board.curr_player)
        
    @abstractmethod       
    def best(self) -> tuple[Move, Node]:
        pass
        
    def calc_best_move(self, max_iter: int = 1000, max_depth = -1):
        max_d = 0
        while len(self.root.untried_actions) > 0:
            board = deepcopy(self.board)
            (move, node) = self.root.expand(board)
            board.make_move(move)
            ev = node.evaluate(board)
            node.backpropagate(ev)
            # node.eval = node.evaluate(self.board)
            max_iter -= 1
        
        for _ in range(max_iter):
            node = self.root
            board = deepcopy(self.board)
            depth = 0
            while len(node.untried_actions) == 0 and not node.is_leaf:
                (move, node) = node.select_child()
                board.make_move(move)
                depth += 1
            
            ev = 0
            if not node.is_leaf:
                if max_depth <= 1 or depth + 1 < max_depth:
                    (move, node) = node.expand(board)
                    board.make_move(move)
                    depth += 1
            ev = node.evaluate(board)
            
            node.backpropagate(ev)
            if depth > max_d:
                max_d = depth
    
    def move(self, move: Move):
        found = False
        for m, node in self.root.children:
            if m == move:
                found = True
                self.root = node
                break
        
        if not found:
            (move, node) = self.root.expand(self.board, move)
            self.root = node
        self.board.make_move(move)
        return
        
    def run(self, input_players: list[str], debug = False, engine_max_iter: int = 1000, engine_max_depth: int = -1):
        while self.board.state == gameState.ONGOING:
            print('___________________________')
            self.calc_best_move(max_iter=engine_max_iter, max_depth=engine_max_depth)
            print("engine calculations: ")
            if debug:
                print("all moves:")
                for move, child in self.root.children:
                    print(f"move {move} has {child.eval}, with {child.visits} visits\n")
            print(f"this position has {self.root.eval} eval")
            (move, node) = self.best()
            print(f"best move is {move} evaluated with {node.eval}")
            
            print(self.board)
            if self.board.curr_player in input_players:
                move = None
                m = input(f"{self.board.curr_player}'s move: \nlegal moves(column number): {self.board.legal_moves}\nEnter your move: ")
                move = self.board.create_move(m)
                
            if move is not None:
                self.move(move)
            else:
                print("ERROR")
                return
                

        print(self.board)
        print('Game over!')
        if self.board.state == gameState.DRAW:
            print('The game ended in a draw!')
        else:
            print(f'{self.board.winner} WON!')