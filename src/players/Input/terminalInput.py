from src.base import Game, Move, gameState, player

class terminalPlayer(player):
    
    def __init__(self, game_board: Game, name: str) -> None:
        super().__init__(game_board, name)

    def get_move(self):
        print(self.board)
        print(f"legal moves: {self.board.legal_moves}")
        return self.board.create_move(input("Enter move: "))
        # self.board.make_move(move)
        
    def move(self, move: Move):
        pass
        