from src.base import Move, Board, gameState, player
from src.players.MCTS.Treeplayer import Node, TreePlayer

import random
import math
import copy


class MCTS_uct_Node(Node):
    """
    A node for a simulation uct tree
    
    Methods:
        * select_child(): Select the child node to explore using uct score
        * expand(board: Board, move: Move = None): Expand the tree by creating a new child node for the given move
        * evaluate(board: Board): Evaluate the node using a random game simulation
    """
    
    def __init__(self, state: Board, parent: "MCTS_uct_Node" = None) -> None:
        super(MCTS_uct_Node, self).__init__(state, parent=parent)
    
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
        new_Node = MCTS_uct_Node(state=board, parent=self)
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
            board.draw()
        elif board.winner == self.player:
            board.win()
        else:
            board.lose()
            
        return board.reward
    
class MCTS_uct_Tree(TreePlayer):
    """
    A tree player using the uct algorithm
    
    Methods:
        * best(): Return the Node with the minimum eval/visits for the next player(what is the move that is the worst for the opponment?)
        * create_node(untried_actions: list[Move], player: player, parent: Node = None) -> Node: Create a new uct node
    """
    
    def __init__(self, game_board: Board, name: str) -> None:
        super(MCTS_uct_Tree, self).__init__(game_board, name)
        
    def best(self):
        return max(self.root.children, key=lambda c: c[1].visits if c[1].visits > 0 else 0)
        # return min(self.root.children, key=lambda c: c[1].eval / c[1].visits if c[1].visits > 0 else 0)
    
    def create_node(self, state: Board, parent: Node = None) -> Node:
        return MCTS_uct_Node(state, parent)

