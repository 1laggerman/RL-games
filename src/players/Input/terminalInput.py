from src.base import Game, Action, gameState, Player

class terminalPlayer(Player):
    
    def __init__(self, game_board: Game, name: str) -> None:
        super().__init__(game_board, name)

    def get_move(self):
        print(self.game)
        print(f"legal moves: {self.game.legal_actions}")
        return self.game.create_action(input("Enter move: "))
        
    def update_state(self, action: Action):
        pass