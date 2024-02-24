from Game import Move, Board, gameState
import random
import math
import copy


class MCTSNode:
    visits: int = 0
    wins: int = 0
    parent: "MCTSNode"
    children: list[tuple[Move, "MCTSNode"]]
    untried_actions: list[Move]
    
    def __init__(self, untried_actions: list[Move], parent: "MCTSNode" = None) -> None:
        self.visits = 0
        self.wins = 0
        self.parent = parent
        self.children = list()
        self.untried_actions = copy.deepcopy(untried_actions)
        
    def select_child_with_epsilon_greedy(self, epsilon: float):
        if random.random() < epsilon:
            return random.choice(self.children)
        return max(self.children, key=lambda c: c[1].wins / c[1].visits if c[1].visits > 0 else 0)
    
    def select_child_with_utc(self):
        assert self.visits > 0
        return max(self.children, key=lambda c: (c[1].wins / c[1].visits if c[1].visits > 0 else 0) + math.sqrt(2) * math.sqrt(math.log(self.visits) / 1 + c[1].visits))
    
    def expand(self, board: Board):
        new_action = self.untried_actions.pop()
        board.make_move(new_action)
        new_Node = MCTSNode(board.legal_moves, self)
        board.unmake_move()
        new_child = (new_action, new_Node)
        self.children.append(new_child)
        return new_child
        
    def simulate(self, board: Board):
        result = 0
        player = board.curr_player
        num_moves = 0
        while board.state is not gameState.DRAW and board.state is not gameState.ENDED:
            choices = [i for i in range(board.legal_moves.__len__())]
            move_idx = random.choice(choices)
            board.make_move(board.legal_moves[move_idx])
            num_moves += 1
        if board.winner == player:
            result = 1
        elif board.state == gameState.DRAW:
            result = 0.5
        self.visits += 1
        self.wins += result
        
        while num_moves > 0:
            board.unmake_move()
            num_moves -= 1
        return result