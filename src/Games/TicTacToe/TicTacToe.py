from src.base import Game, Action, gameState, Piece, Role, Move
import os
import numpy as np

class TicTacToe_Action(Action):
        
    def __init__(self, name: str, action_taker: Role) -> None:

        affects = []
        loc = name.split(",") # get piece location
        added_p = Piece(action_taker.name, action_taker) # create a new piece
        
        move = Move(added_p, (int(loc[0]), int(loc[1]))) # place the piece on the correct squere
        affects.append(move) # add the move to the list of affects

        super(TicTacToe_Action, self).__init__(name, affects)

        
    def __eq__(self, __value: 'TicTacToe_Action') -> bool:
        if isinstance(__value, TicTacToe_Action):
            return self.name == __value.name
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
    
    legal_actions: list[TicTacToe_Action]
    
    def __init__(self) -> None:
        super(TicTacToe_Game, self).__init__((3, 3), [Role('X'), Role('O')])
        self.legal_actions = [TicTacToe_Action(f"{i},{j}", self.roles[0]) for i in range(3) for j in range(3)]
        self.all_actions = self.legal_actions.copy()
    
    def create_action(self, input: str) -> Action:
        action = TicTacToe_Action(input, self.curr_role)
        if action not in self.legal_actions:
            raise ValueError("Illegal action")
        return action
    
    def update_state(self, last_action: TicTacToe_Action):

        move = last_action.affects[0]

        x = move.dest_location[0]
        y = move.dest_location[1]
        
        p = move.moved_piece

        self.curr_role.pieces.append(p)
        self.board[(x, y)] = p

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

            while test_loc[0] >= 0 and test_loc[0] < self.board.shape[0] and test_loc[1] >= 0 and test_loc[1] < self.board.shape[1]:
                location: Piece = self.board[loc[0] + i * dx, loc[1] + i * dy]
                if location is None or location.name != self.curr_role.name:
                    break
                i += 1
                test_loc = (loc[0] + i * dx, loc[1] + i * dy)

            if i == 3: # 3 in a row
                return self.win()
        
        if len(self.legal_actions) == 0:
            return self.draw()
        
        return []
            
    def reverse_state(self, action: TicTacToe_Action):
        self.board[action.affects[0].dest_location] = None
        self.legal_actions.append(action)
        self.prev_player()
        for action in self.legal_actions:
            action.action_taker = self.curr_role
        self.next_player()
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
    
    def encode(self):
        encoded_state = np.stack(
            (self.board == self.roles[0], self.board == None, self.board == self.roles[1])
        ).astype(np.float32)
        return encoded_state
