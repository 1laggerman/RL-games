from src.base import Game, Move, gameState, Player

class terminalPlayer(Player):
    
    def __init__(self, game_board: Game, name: str) -> None:
        super().__init__(game_board, name)

    def get_move(self):
        print(self.game)
        print(f"legal moves: {self.game.legal_moves}")
        return self.game.create_move(input("Enter move: "))
        # self.board.make_move(move)
        
    def update_state(self, move: Move):
        if move.reward is not None:
            self.recv_reward(move.reward)