from src.base import Game, Move, gameState, Player, bind


def play(game: 'Game', players: list['Player']):
    """
    simulates a simple game loop between any 2 players

    Args:
    -----
        * game (Game): the game that is being played
        * players (list[player]): the players playing the game
    """
    bind(game, players)
    while game.state == gameState.ONGOING:
        move = game.curr_role.get_move()
        if move is None:
            print("Invalid move")
            return
        game.make_move(move)
        game.alert_players(move)
        
    if game.state == gameState.ENDED:
        print(f"Winner: {game.winner.name}")
        print(f"Reward: {game.reward}")
    else:
        print("Draw")