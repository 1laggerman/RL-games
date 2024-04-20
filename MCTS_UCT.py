from Game import Move, Board, gameState
from Tree import Node, SearchTree
from enum import Enum
import random
import math
import copy


class MCTS_uct_Node(Node):
    
    def __init__(self, untried_actions: list[Move], player: str, parent: "MCTS_uct_Node" = None) -> None:
        super(MCTS_uct_Node, self).__init__(untried_actions=untried_actions, player=player, parent=parent)
    
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
        new_Node = MCTS_uct_Node(copy.deepcopy(board.legal_moves), board.curr_player, self)
        if board.state == gameState.ENDED or board.state == gameState.DRAW:
            new_Node.is_leaf = True
            board.next_player()
            new_Node.player = board.curr_player
            board.prev_player()
        board.unmake_move()
        new_child = (new_action, new_Node)
        self.children.append(new_child)
        return new_child
        
    def evaluate(self, board: Board):
        while board.state is not gameState.DRAW and board.state is not gameState.ENDED:
            choices = [i for i in range(board.legal_moves.__len__())]
            move_idx = random.choice(choices)
            board.make_move(board.legal_moves[move_idx])
        
        if board.state == gameState.DRAW:
            return 0.5
        elif board.winner == self.player:
            return 1
        return 0
    
class MCTS_uct_Tree(SearchTree):
    
    def __init__(self, game_board: Board) -> None:
        super(MCTS_uct_Tree, self).__init__(game_board)
        self.root = MCTS_uct_Node(copy.deepcopy(game_board.legal_moves), player=game_board.curr_player)
        
    def best(self):
        return min(self.root.children, key=lambda c: c[1].eval / c[1].visits if c[1].visits > 0 else 0)

