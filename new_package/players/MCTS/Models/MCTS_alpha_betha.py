from new_package.base import Move, Board, gameState
from new_package.players.MCTS.Treeplayer import Node, TreePlayer

class MCTS_ab_Node(Node):
    
    def __init__(self, untried_actions: list[Move], player: str, parent: Node = None) -> None:
        super().__init__(untried_actions, player, parent)
        self.tree_eval = 0
        self.alpha = float('-inf')
        self.beta = float('inf')
        
    def backpropagate(self, eval: float):
        self.visits += 1
        
        self.tree_eval += eval
        if eval > self.alpha:
            self.alpha = eval
        if eval < self.beta:
            self.beta = eval
        if self.parent:
            self.parent.backpropagate(eval)