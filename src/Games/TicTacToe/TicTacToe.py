from src.base import Game, Action, gameState, Piece, Role, Move
import os
import numpy as np

class TicTacToe_Action(Action):
        
    def __init__(self, name: str, action_taker: Role) -> None:
        super(TicTacToe_Action, self).__init__(name, action_taker, [])

        locs = name.split(",") # get piece location
        added_p = Piece(self.name, action_taker) # create a new piece
        move = Move(added_p, (int(locs[0]), int(locs[1]))) # place the piece on the correct squere
        self.affects.append(move) # add the move to the list of affects

        
    def __eq__(self, __value: 'TicTacToe_Action') -> bool:
        if isinstance(__value, TicTacToe_Action):
            return self.name == __value.name and self.action_taker == __value.action_taker
        return False
        

class TicTacToe_Game(Game):
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
    
    legal_moves: list[TicTacToe_Action]
    
    def __init__(self) -> None:
        super(TicTacToe_Game, self).__init__((3, 3), [Role('X'), Role('O')])
        self.legal_moves = [TicTacToe_Action(f"{i},{j}") for i in range(3) for j in range(3)]
        self.all_actions = self.legal_moves.copy()
    
    def create_action(self, input: str) -> Action:
        try:
            action = TicTacToe_Action(input)
        except:
            pass
        return None
    
    def update_state(self, last_action: TicTacToe_Action):

        move = last_action.affects[0]

        x = move.dest_location[0]
        y = move.dest_location[1]
        
        p = Piece(self.curr_role.name, self.curr_role, location=last_action.dest_location)

        self.curr_role.pieces.append(p)
        self.board[move.dest_location] = p

        self.legal_actions.remove(last_action)

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
            test_loc = (loc[0], loc[1])
            move = TicTacToe_Action(f"{loc[0]},{loc[1]}")
            while test_loc[0] >= 0 and test_loc[0] < self.board.shape[0] and test_loc[1] >= 0 and test_loc[1] < self.board.shape[1]:
                location: Piece = self.board[loc[0] + i * dx, loc[1] + i * dy]
                if location is None or location.name != self.curr_role.name:
                    break
                i += 1
                test_loc = (loc[0] + i * dx, loc[1] + i * dy)

            if i == 3: # 3 in a row
                self.win()
                return
        
        if len(self.legal_actions) == 0:
            self.draw()
            
    def reverse_state(self, action: TicTacToe_Action):
        self.board[action.affects[0].dest_location] = None
        self.legal_actions.append(action)
        self.state = gameState.ONGOING
        self.winner = None
        self.reward = 0
        for role in self.roles:
            role.reward = 0
        
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
    
    def encode(self):
        encoded_state = np.stack(
            (self.board == self.roles[0], self.board == None, self.board == self.roles[1])
        ).astype(np.float32)
        return encoded_state
