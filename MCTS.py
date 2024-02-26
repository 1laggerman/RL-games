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
    wins: int = 0
    parent: "MCTSNode"
    player: str
    children: list[tuple[Move, "MCTSNode"]]
    untried_actions: list[Move]
    is_leaf: bool = False
    
    def __init__(self, untried_actions: list[Move], player: str, parent: "MCTSNode" = None) -> None:
        self.visits = 0
        self.wins = 0
        self.parent = parent
        self.children = list()
        self.player = player
        self.untried_actions = copy.deepcopy(untried_actions)
        self.is_leaf = False
        if len(self.untried_actions) == 0:
            self.is_leaf = True
        
    def select_child_with_epsilon_greedy(self, epsilon: float):
        if random.random() < epsilon:
            return random.choice(self.children)
        return min(self.children, key=lambda c: c[1].wins / c[1].visits if c[1].visits > 0 else 0)
    
    def select_child_with_utc(self):
        assert self.visits > 0
        return min(self.children, key=lambda c: (c[1].wins / c[1].visits if c[1].visits > 0 else 0) + math.sqrt(2) * math.sqrt(math.log(self.visits) / 1 + c[1].visits))        
            
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
        
    def simulate(self, board: Board):
        while board.state is not gameState.DRAW and board.state is not gameState.ENDED:
            choices = [i for i in range(board.legal_moves.__len__())]
            move_idx = random.choice(choices)
            board.make_move(board.legal_moves[move_idx])
        
        return board.winner
    
    def backpropagate(self, winner: str):
        self.visits += 1
        if self.player == winner:
            self.wins += 1
        elif winner == "":
            self.wins += 0.5
            
        if self.parent:
            self.parent.backpropagate(winner)
            
class MCTSTree():
    root: MCTSNode
    board: Board
    
    def __init__(self, game_board: Board) -> None:
        self.board = game_board
        self.root = MCTSNode(copy.deepcopy(game_board.legal_moves), player=game_board.curr_player)
        
    def calc_best_move(self, max_iter: int = 3000, max_depth = -1, alg: ALGORITHMS = ALGORITHMS.UCT, *alg_params):
        max_d = 0
        for _ in range(max_iter):
            node = self.root
            board = copy.deepcopy(self.board)
            depth = 0
            while len(node.untried_actions) == 0 and not node.is_leaf:
                if alg == ALGORITHMS.EPSILON_GREEDY:
                    (move, node) = node.select_child_with_epsilon_greedy(float(alg_params[0]))
                elif alg == ALGORITHMS.UCT:
                    (move, node) = node.select_child_with_utc()
                board.make_move(move)
                depth += 1
            winner = ""
            if not node.is_leaf:
                if max_depth <= 1 or depth + 1 < max_depth:
                    (move, node) = node.expand(board)
                    board.make_move(move)
                    depth += 1
                winner = node.simulate(copy.deepcopy(board))
            else:
                winner = board.winner
            node.backpropagate(winner)
            if depth > max_d:
                max_d = depth
        print("reached depth: ", depth)
            
            
    def best(self):
        return min(self.root.children, key=lambda c: c[1].wins / c[1].visits if c[1].visits > 0 else 0)
    
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
        
    def run(self, input_players: list[str], alg: ALGORITHMS = ALGORITHMS.UCT, *alg_params):
        while self.board.state == gameState.ONGOING:
            print(self.board)
            print("engine calculations: ")
            print("wins: ", self.root.wins, " visits: ", self.root.visits)
            move = None
            if self.board.curr_player in input_players:
                m = input(f"{self.board.curr_player}'s move: \nlegal moves(column number): {self.board.legal_moves}\nEnter your move: ")
                move = self.board.create_move(m)
            else: 
                self.calc_best_move(max_iter=1000, max_depth=-1, alg=alg, *alg_params)
                (move, node) = self.best()
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