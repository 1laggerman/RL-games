from Game import Board, Move, gameState
import numpy as np

class connect4_Move(Move):    
    location: int
    
    def __init__(self, name: str, player: str, loc: int) -> None:
        if player == "":
            player = " "
        super().__init__(name, player)
        self.location = loc
    
    def __eq__(self, __value: "connect4_Move") -> bool:
        return super().__eq__(__value) and self.location == __value.location
    
    def __str__(self) -> str:
        return f"{self.location}"
    
    def __repr__(self):
        return str(self)
        

class connect4_Board(Board):
    
    cols_heights: list[int]
    legal_moves: list[connect4_Move]
    
    def __init__(self, rows: int, cols: int, players: list[str] = ['X', 'O']) -> None:
        super().__init__((rows, cols), players)
        self.legal_moves = [connect4_Move(f"{i}", " ", loc=i) for i in range(cols)]
        self.cols_heights = [0 for i in range(cols)]
    
    def make_move(self, move: connect4_Move):
        temp_player = move.player
        move.player = " "
        if move not in self.legal_moves:
            move.player = self.curr_player
            print("illegal move")
            return
        move.player = self.curr_player
        
        self.history.append(move)
        
        self.board[self.cols_heights[move.location], move.location] = move.player
        self.cols_heights[move.location] = self.cols_heights[move.location] + 1
        
        if self.cols_heights[move.location] == self.rows():
            self.legal_moves.remove(connect4_Move(move.name, "", move.location))
            
        self.check_state(move)
        self.next_player()
    
    def unmake_move(self, move: connect4_Move = None):
        if move == None:
            move = self.history.pop()
        self.board[self.cols_heights[move.location] - 1, move.location] = " "
        if self.cols_heights[move.location] == self.rows():
            self.legal_moves.insert(move.location, connect4_Move(move.name, "", move.location))
        self.cols_heights[move.location] = self.cols_heights[move.location] - 1
        
        self.winner = ""
        self.state = gameState.ONGOING
        self.prev_player()
    
    def __str__(self) -> str:
        result = ""
        for row in reversed(self.board):
            result += "|"
            for cell in row:
                result += cell + '|'
            result += "\n"
        result += '-'
        result += "--" * (self.cols()) + "\n"
        result += " "
        result += " ".join(str(i) for i in range(self.cols()))
        return result
    
    def __repr__(self):
        return str(self)
    
    def rows(self):
        return self.board.shape[0]
    
    def cols(self):
        return self.board.shape[1]
    
    def check_state(self, last_move: connect4_Move):
        y = last_move.location
        x = self.cols_heights[last_move.location] - 1
        
        player = self.curr_player
        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]

        for dx, dy in directions:
            count = 1  # Start counting from the current piece
        
            # Count in the positive direction
            i = 1
            while 0 <= x + i * dx < self.rows() and 0 <= y + i * dy < self.cols() and self.board[x + i * dx][y + i * dy] == player:
                count += 1
                i += 1
    
            # Count in the negative direction
            i = 1
            while 0 <= x - i * dx < self.rows() and 0 <= y - i * dy < self.cols() and self.board[x - i * dx][y - i * dy] == player:
                count += 1
                i += 1

            if count >= 4:
                self.state = gameState.ENDED
                self.winner = player
                return player  # Win detected
        
        if self.legal_moves.__len__() == 0:
            self.state = gameState.DRAW

        return gameState.ONGOING
        