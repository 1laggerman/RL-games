from Game import Move, Board, gameState
from enum import Enum
import random
import math
import copy

class ALGORITHMS(Enum):
    EPSILON_GREEDY = 1
    UCT = 0


class MCTSNode:
    visits: int = 0
    eval: int = 0
    parent: "MCTSNode"
    player: str
    children: list[tuple[Move, "MCTSNode"]]
    untried_actions: list[Move]
    is_leaf: bool = False
    
    def __init__(self, untried_actions: list[Move], player: str, parent: "MCTSNode" = None) -> None:
        self.visits = 0
        self.eval = 0
        self.parent = parent
        self.children = list()
        self.player = player
        self.untried_actions = copy.deepcopy(untried_actions)
        self.is_leaf = False
        if len(self.untried_actions) == 0:
            self.is_leaf = True
    
    def select_child(self):
        
        assert self.visits > 0, "Parent node has zero visits."
        # Exploration parameter
        C = math.sqrt(2)

        # Calculate UCT score for each child and select the child with the highest score
        best_score = float('-inf')
        best_child = None
        for child in self.children:
            # exploiting 1 - (wins / visits) because the child node is a different player
            exploitation = (child[1].visits - child[1].eval) / child[1].visits if child[1].visits > 0 else 0
            exploration = math.sqrt(math.log(self.visits) / (1 + child[1].visits))
            uct_score = exploitation + C * exploration

            if uct_score > best_score:
                best_score = uct_score
                best_child = child

        return best_child
           
    def expand(self, board: Board, move: Move = None):
        new_action = move
        if move is None or move not in self.untried_actions:
            new_action = self.untried_actions.pop()
        
        board.make_move(new_action)
        new_Node = MCTSNode(copy.deepcopy(board.legal_moves), board.curr_player, self)
        if board.state == gameState.ENDED or board.state == gameState.DRAW:
            new_Node.is_leaf = True
        board.unmake_move()
        new_child = (new_action, new_Node)
        self.children.append(new_child)
        return new_child
        
    def evaluate(self, board: Board):
        curr = board.curr_player
        while board.state is not gameState.DRAW and board.state is not gameState.ENDED:
            choices = [i for i in range(board.legal_moves.__len__())]
            move_idx = random.choice(choices)
            board.make_move(board.legal_moves[move_idx])
        
        if board.state == gameState.DRAW:
            return 0.5
        elif board.winner == curr:
            return 1
        return 0
    
    def backpropagate(self, eval: int):
        self.visits += 1
        
        self.eval += eval
            
        if self.parent:
            self.parent.backpropagate(1-eval)
            
class MCTSTree():
    root: MCTSNode
    board: Board
    
    def __init__(self, game_board: Board) -> None:
        self.board = game_board
        self.root = MCTSNode(copy.deepcopy(game_board.legal_moves), player=game_board.curr_player)
        
    def calc_best_move(self, max_iter: int = 1000, max_depth = -1):
        max_d = 0
        for _ in range(max_iter):
            node = self.root
            board = copy.deepcopy(self.board)
            depth = 0
            while len(node.untried_actions) == 0 and not node.is_leaf:
                (move, node) = node.select_child()
                board.make_move(move)
                depth += 1
                
            score = 0
            if not node.is_leaf:
                if max_depth <= 1 or depth + 1 < max_depth:
                    (move, node) = node.expand(board)
                    board.make_move(move)
                    depth += 1
            score = node.evaluate(board)
            
            node.backpropagate(score)
            if depth > max_d:
                max_d = depth
        print("reached depth: ", depth)
            
            
    def best(self):
        return min(self.root.children, key=lambda c: c[1].eval / c[1].visits if c[1].visits > 0 else 0)
    
    def move(self, move: Move):
        found = False
        for m, node in self.root.children:
            if m == move:
                found = True
                self.root = node
                break
        
        if not found:
            (m, node) = self.root.expand(self.board, move)
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
                    print(f"move {move} has {child.visits - child.eval} / {child.visits}\n")
            print(f"this position has {self.root.eval} / {self.root.visits} wins")
            (move, node) = self.best()
            print(f"best move is {move} evaluated with {node.visits - node.eval} / {node.visits}")
            
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