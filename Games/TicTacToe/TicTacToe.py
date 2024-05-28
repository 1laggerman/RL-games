from Games.Game import Board, Move, gameState

class TicTacToe_move(Move):
    location: tuple[int, int]
        
    def __init__(self, name: str) -> None:
        super(TicTacToe_move, self).__init__(name)
        locs = name.split(",")
        self.location[0] = int(locs[0])
        self.location[1] = int(locs[1])
        

class TicTacToe_Board(Board):
    
    legal_moves: list[TicTacToe_move]
    
    def __init__(self, board_size: tuple, players: list[str] = ['X', 'O']) -> None:
        super(TicTacToe_Board, self).__init__(board_size, players)
        self.legal_moves = [TicTacToe_move(f"{i},{j}") for i in range(board_size[0]) for j in range(board_size[1])]

    
    def create_move(self, input: str) -> Move:
        try:
            return TicTacToe_move(input)
        except:
            pass
        return None
    
    def make_move(self, move: TicTacToe_move):
        self.board[move.location[0], move.location[1]] = self.curr_player
        self.legal_moves.remove(move)
    
    def unmake_move(self, move: TicTacToe_move):
        self.board[move.location[0], move.location[1]] = ' '
        self.legal_moves.append(move)
    
    def update_state(self, last_move: TicTacToe_move):
        y = last_move.location[0]
        x = last_move.location[1]
        
        player = self.curr_player
        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]

        for dx, dy in directions:
            loc = (0, 0)
            if dx == 0:
                loc[0] = x
            if dy == 0:
                loc[1] = y
            if dy == -1:
                loc[1] = self.board.shape[1] - 1
                
            i = 1
            while loc[0] + i * dx < self.board.shape[0] and loc[1] + i * dy < self.board.shape[1]:
                if self.board[x + i * dx, y + i * dy] != player:
                    break
                i += 1

            # win detected
            self.win()
        