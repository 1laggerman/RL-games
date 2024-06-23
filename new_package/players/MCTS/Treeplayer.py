from new_package.base import Move, Board, gameState, player

import random
import math
from copy import deepcopy
from abc import ABC, abstractmethod
import tqdm
import time

class Node(ABC):
    visits: int = 0
    eval: float = 0
    parent: 'Node'
    player: 'player'
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
            
class TreePlayer(player, ABC):
    root: Node
    board: Board
    
    def __init__(self, game_board: Board, name: str) -> None:
        super(TreePlayer, self).__init__(game_board, name)
        
        self.root = None
        # self.board = game_board
        # self.root = self.create_node(game_board.legal_moves, player=game_board.curr_player)
        # self.root = Node(game_board.legal_moves, player=game_board.curr_player)
        
    def get_move(self):
        if self.root == None:
            self.root = self.create_node(self.board.legal_moves, player=self.board.curr_player, parent=None)
        
        self.calc_best_move(max_iter=100, max_depth=-1)
        return self.best()[0]
        
    @abstractmethod       
    def best(self) -> tuple[Move, Node]:
        pass

    @abstractmethod
    def create_node(self, untried_actions: list[Move], player: player, parent: Node = None) -> Node:
        pass
        
    def calc_best_move(self, max_iter: int = 1000, max_depth = -1):
        max_d = 0
        while len(self.root.untried_actions) > 0:
            board = deepcopy(self.board)
            (move, node) = self.root.expand(board)
            board.make_move(move)
            # ev = node.evaluate(board)
            # self.board.unmake_move()
            node.eval = node.evaluate(board)
            node.backpropagate(node.eval)
            max_iter -= 1
        
        # with tqdm.tqdm(total=max_iter)
        for _ in tqdm.tqdm(range(max_iter)):
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
                if max_depth <= 1 or depth + 1 < max_depth:
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