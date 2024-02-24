from Game import Board, Move
import numpy as np

class connect4_Move(Move):    
    location: int
    
    def __init__(self, name: str, player: str, loc: int) -> None:
        super().__init__(name, player)
        self.location = loc

class connect4_Board(Board):
    
    cols_capacity: list[int]
    
    def __init__(self, rows: int, cols: int) -> None:
        super().__init__((rows, cols))
        self.legal_moves = [connect4_Move("0", "", loc=0) for i in cols]
        self.cols_capacity = [0 for i in cols]
    
    def make_move(self, move: connect4_Move):
        self.state[self.cols_capacity[move.location], move.location] = move.player
        self.cols_capacity[move.location + 1]
        
        # if self.cols_capacity[move.location] == self.state.shape[0] - 1:
            # self.legal_moves.remove(conn)
        
    
    def unmake_move(self, move: connect4_Move):
        pass