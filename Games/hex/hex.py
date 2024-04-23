from Games.Game import Board, Move, gameState
import numpy as np
import itertools as it
from collections import deque
from copy import deepcopy

class hex_Move(Move):    
    location: tuple[int, int]
    
    def __init__(self, name: str, loc: tuple[int, int]) -> None:
        super(hex_Move, self).__init__(name)
        self.location = loc
    
    def __eq__(self, value: "hex_Move") -> bool:
        return super(hex_Move, self).__eq__(value) and self.location[0] == value.location[0] and self.location[1] == value.location[1]
    
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
    linked_to_edge: np.ndarray
    
    def __init__(self, size: int, players: list[str] = ['X', 'O']) -> None:
        super(hex_Board, self).__init__((size, size), players)
        self.legal_moves = [hex_Move(f"{i} {j}", loc=(i, j)) for i, j in it.product(range(size), repeat=2)]
        self.linked_to_edge = np.zeros((players.__len__(), 2, size, size), dtype=bool)
        
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
        
        self.board[*move.location] = self.curr_player
        
        self.legal_moves.remove(move)
            
        self.update_state(move)
        if self.state == gameState.ONGOING:
            self.next_player()
    
    def update_state(self, move: hex_Move):
        links = self.get_links(move)
        for link in links:
            if self.linked_to_edge[self.curr_player_idx, 0, *link.location]:
                self.linked_to_edge[self.curr_player_idx, 0, *move.location] = True
            if self.linked_to_edge[self.curr_player_idx, 1, *link.location]:
                self.linked_to_edge[self.curr_player_idx, 1, *move.location] = True
        
        if np.all(self.linked_to_edge[self.curr_player_idx, :, *move.location]):
            self.state = gameState.ENDED
            self.winner = self.curr_player
            return
            
        if move.location[self.curr_player_idx] == 0:
            self.linked_to_edge[self.curr_player_idx, 0, *move.location] = True
        elif move.location[self.curr_player_idx] == self.board.shape[self.curr_player_idx] - 1:
            self.linked_to_edge[self.curr_player_idx, 1, *move.location] = True    
            
        found_edges = self.linked_to_edge[self.curr_player_idx, :, *move.location]
        for i, edge in enumerate(found_edges):
            if edge == True:
                self.dynamic_BFS(move, i)
        

    def dynamic_BFS(self, root: Move, edge: int):
        q = deque([root])
        while q.__len__() > 0:
            m = q.popleft()
            for link in self.get_links(m):
                if self.linked_to_edge[self.curr_player_idx, edge, *link.location] == False:
                    self.linked_to_edge[self.curr_player_idx, edge, *link.location] = True
                    
                    if np.all(self.linked_to_edge[self.curr_player_idx, :, *link.location]):
                        self.state = gameState.ENDED
                        self.winner = self.curr_player
                        return
                    
                    q.append(link)
            
    
    def get_links(self, move: hex_Move) -> list[hex_Move]:
        neighbors = ((-1, 1), (0, 1), (1, 0))
        links = []
        for n in neighbors:
            link = move + n
            if link.location[0] >= 0 and link.location[1] >= 0 and link.location[0] < self.board.shape[1] and link.location[1] < self.board.shape[0]:
                if self.board[link.location] == self.curr_player:
                    links.append(move + n)
            link = move - n
            if link.location[0] >= 0 and link.location[1] >= 0 and link.location[0] < self.board.shape[1] and link.location[1] < self.board.shape[0]:
                if self.board[link.location] == self.curr_player:
                    links.append(move - n)
        return links
    
    def unmake_move(self, move: hex_Move = None):
        if move is None:
            move = self.history.pop()
        self.board[*move.location] = str(' ')
        self.legal_moves.append(move)
        
        if self.state != gameState.ENDED:
            self.prev_player()
        
        self.winner = ""
        self.state = gameState.ONGOING
        
    def encode(self):
        board = deepcopy(self.board)
        player_states = np.array([board == player for player in self.players])
        empty_state = self.board == ' '
        enc: np.ndarray = np.concatenate([player_states, empty_state.reshape((1, *empty_state.shape))])
        return enc.astype(np.float32)

    def __str__(self):
        board_str = ""
        rows, cols = self.board.shape
        indent = 0
        headings = " "*5+(" "*3).join([str(i) for i in range(cols)])
        # print(headings)
        board_str += headings + "\n"
        tops = " "*5+(" "*3).join("-"*cols)
        # print(tops)
        board_str += tops + "\n"
        roof = " "*4+"/ \\"+"_/ \\"*(cols-1)
        # print(roof)
        board_str += roof + "\n"
        # color_mapping = lambda i : " WB"[i]
        for r in range(rows):
            row_mid = " " * (indent - (len(str(r)) - 1)) 
            row_mid += " {} | ".format(r)
            row_mid += " | ".join(self.board[r, :])
            row_mid += " | {} ".format(r)
            # print(row_mid)
            board_str += row_mid + "\n"
            row_bottom = " "*indent
            row_bottom += " "*3+" \\_/"*cols
            if r<rows-1:
                row_bottom += " \\"
            # print(row_bottom)
            board_str += row_bottom + "\n"
            # print(str(r).__len__())
            indent += 3 - str(r).__len__()
        headings = " "*(indent-(3 - str(r).__len__())) + headings
        # print(headings)
        board_str += headings + "\n"
        return board_str
    
    def __repr__(self) -> str:
        return str(self)

