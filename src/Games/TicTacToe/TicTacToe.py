from src.base import Board, Move, gameState, player, Piece

class TicTacToe_move(Move):
    """
    A move for a TicTacToe game.
    
    Attributes:
        * location (tuple[int, int]): The location of the move on the board.
        * name (str): The name of the move. format: "x,y"
    """
    location: tuple[int, int]
        
    def __init__(self, name: str) -> None:
        super(TicTacToe_move, self).__init__(name)
        locs = name.split(",")
        self.location = (int(locs[0]), int(locs[1]))
        # self.location[1] = int(locs[1])
        
    def __eq__(self, __value: 'TicTacToe_move') -> bool:
        return self.location[0] == __value.location[0] and self.location[1] == __value.location[1]
        

class TicTacToe_Board(Board):
    """
    A TicTacToe game board.

    Attributes:
        Same as super class(Board)

    Methods:
        update_state(last_move: TicTacToe_move): Updates the board state after a move was made.
        reverse_state(): Reverses the board state.
        create_move(input: str): Creates a TicTacToe_move from a string with format "x,y".
        __str__(): Returns a string representation of the board state to draw the board in the terminal.
    """
    
    legal_moves: list[TicTacToe_move]
    
    def __init__(self, board_size: tuple, players: list = []) -> None:
        super(TicTacToe_Board, self).__init__(board_size, players)
        self.legal_moves = [TicTacToe_move(f"{i},{j}") for i in range(board_size[0]) for j in range(board_size[1])]
    
    def create_move(self, input: str) -> Move:
        try:
            return TicTacToe_move(input)
        except:
            pass
        return None
    
    def update_state(self, last_move: TicTacToe_move):
        y = last_move.location[0]
        x = last_move.location[1]
        
        self.board[x, y] = Piece(self.curr_player.name, self.curr_player, location=(x, y))
        self.legal_moves.remove(last_move)
        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]

        for dx, dy in directions:
            loc = [0, 0]
            if dx == 0:
                loc[0] = x
            if dy == 0:
                loc[1] = y
            if dy == -1:
                loc[1] = self.board.shape[1] - 1
                
            i = 0
            move = TicTacToe_move(f"{loc[0]},{loc[1]}")
            while move.location[0] >= 0 and move.location[0] < self.board.shape[0] and move.location[1] >= 0 and move.location[1] < self.board.shape[1]:
                location: Piece = self.board[loc[0] + i * dx, loc[1] + i * dy]
                if location is None or location.name != self.curr_player.name:
                    break
                i += 1
                move.location = (loc[0] + i * dx, loc[1] + i * dy)

            if i == 3: # 3 in a row
                self.win()
                return
        
        if len(self.legal_moves) == 0:
            self.draw()
            return
        
        return self
            
    def reverse_state(self, move: TicTacToe_move):
        y = move.location[0]
        x = move.location[1]
        self.board[x, y] = None
        self.legal_moves.append(move)
        self.state = gameState.ONGOING
        self.winner = None
        
    def __str__(self):
        board_str = ''
        for i in range(self.board.shape[0]):
            for j in range(self.board.shape[1]):
                if self.board[i, j] == None:
                    board_str += '   '
                else:
                    board_str += ' ' + str(self.board[i, j]) + ' '
                if j < self.board.shape[1] - 1:
                    board_str += '|'
            board_str += '\n'
            
            if i < self.board.shape[0] - 1:
                dots = 4 * (self.board.shape[1]) - 1
                board_str += '-' * dots + '\n'

        return board_str
    
        