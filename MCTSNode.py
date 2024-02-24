from Game import Move, Board, gameState
import random


class MCTSNode:
    visits: int = 0
    wins: int = 0
    parent: "MCTSNode"
    children: list[tuple[Move, "MCTSNode"]]
    
    def __init__(self, parent: "MCTSNode" = None) -> None:
        self.visits = 0
        self.wins = 0
        self.parent = parent
        self.children = list()
        
    def simulate(self, board: Board):
        result = 0
        self.visits += 1
        player = board.curr_player
        num_moves = 0
        while board.state is not gameState.DRAW and board.state is not gameState.ENDED:
            choices = [i for i in range(board.legal_moves.__len__())]
            move_idx = random.choice(choices)
            board.make_move(board.legal_moves[move_idx])
            num_moves += 1
        if board.winner == player:
            self.wins += 1
        elif board.state == gameState.DRAW:
            self.wins += 0.5
        
        for i in range(num_moves):
            board.unmake_move()