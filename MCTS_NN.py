from Game import Move, Board, gameState
from Tree import Node, SearchTree
import math
import copy

import torch

class value_head(torch.nn.Module):
    def __init__(self, in_features) -> None:
        super().__init__()
        self.fc1 = torch.nn.Linear(in_features, 32)
        self.fc2 = torch.nn.Linear(32, 1)
        
    def forward(self, x: torch.Tensor):
        in_features = x.size(1) * x.size(2)
        x = x.view(-1, in_features)
        x = torch.nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class MCTS_NN_Node(Node):
    
    def __init__(self, board_size: int, untried_actions: list[Move], player: str, parent: "MCTS_NN_Node" = None) -> None:
        super(MCTS_NN_Node, self).__init__(untried_actions=untried_actions, player=player, parent=parent)
        self.vh = value_head(board_size)
        self.crit = torch.nn.MSELoss()
        self.opt = torch.optim.Adam(self.vh.parameters(), lr=0.1, betas=(0.9, 0.999), weight_decay=0.1)
        
    
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
        new_Node = MCTS_NN_Node(copy.deepcopy(board.legal_moves), board.curr_player, self)
        if board.state == gameState.ENDED or board.state == gameState.DRAW:
            new_Node.is_leaf = True
        board.unmake_move()
        new_child = (new_action, new_Node)
        self.children.append(new_child)
        return new_child
        
    def evaluate(self, board: Board):
        return self.vh.forward(torch.Tensor(board.board))
    
class MCTS_NN_Tree(SearchTree):
    
    def __init__(self, game_board: Board) -> None:
        super(MCTS_NN_Tree, self).__init__(game_board)
        self.root = MCTS_NN_Node(copy.deepcopy(game_board.legal_moves), player=game_board.curr_player, board_size=game_board.board.size)
        
    def best(self):
        return min(self.root.children, key=lambda c: c[1].eval / c[1].visits if c[1].visits > 0 else 0)
