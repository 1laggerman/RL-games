from Game import Board, Move, gameState
import numpy as np
import itertools as it

class hex_Move(Move):    
    location: tuple[int, int]
    
    def __init__(self, name: str, loc: tuple[int, int]) -> None:
        super().__init__(name)
        self.location = loc
    
    def __eq__(self, __value: "hex_Move") -> bool:
        return super().__eq__(__value) and self.location[0] == __value.location[0] and self.location[1] == self.location[1]
    
    def __add__(self, other: tuple[int, int]):
        new_loc = (self.location[0] + other[0], self.location[1] + other[1])
        return hex_Move(f"{new_loc[0]} {new_loc[1]}", new_loc)
    
    def __sub__(self, other: tuple[int, int]):
        new_loc = (self.location[0] - other[0], self.location[1] - other[1])
        return hex_Move(f"{new_loc[0]} {new_loc[1]}", new_loc)
    
    def __str__(self) -> str:
        return f"{self.location}"
    
    def __repr__(self):
        return str(self)
    
class hex_Board(Board):
    legal_moves: list[hex_Move]
    
    def __init__(self, size: int, players: list[str] = ['X', 'O']) -> None:
        super().__init__((size, size), players)
        self.legal_moves = [hex_Move(f"({i}, {j})", loc=i) for i, j in it.product(range(size), repeat=2)]
        
    def create_move(self, input: str) -> Move:
        try:
            move = hex_Move(input, tuple(int(num) for num in input.split(' ')))
            if self.is_legal_move(move):
                return move
        except:
            pass
        return None  
        
    def make_move(self, move: hex_Move):
        self.history.append(move)
        
        self.board[move.location[0], move.location[1]] = self.curr_player
        
        # update legal moves
        self.legal_moves.remove(move)
            
        self.update_state(move)
        self.next_player()
        
    
    def update_state(self, move: hex_Move):
        # TODO: write
        # check end-game
        # TODO: add matrix to check the connection to the edges of the board as a tuple to both the edges.
        pass
    
    def get_links(self, move: hex_Move) -> list[hex_Move]:
        neighbors = ((1, 1), (0, 1), (1, 0))
        links = []
        for n in neighbors:
            link = move + n
            if link.location[0] >= 0 and link.location[1] >= 0 and link.location[0] < self.board.shape[1] and link.location[1] < self.board.shape[0]:
                links.append(move + n)
            link = move - n
            if link.location[0] >= 0 and link.location[1] >= 0 and link.location[0] < self.board.shape[1] and link.location[1] < self.board.shape[0]:
                links.append(move - n)
        return links
    
    def unmake_move(self, move: hex_Move = None):
        pass
