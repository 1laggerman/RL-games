from package.Games.Game import Board, Move, gameState
from package.Engines.player import player

# def next_player()

def bind(board: Board, players: list['player']):
    if len(players) == 0:
        print("No players")
        return
    
    for player in players:
        player.board = board
        board.players = players

def play(board: Board, players: list['player']):
    bind(board, players)
    while board.state != gameState.ONGOING:
        move = board.curr_player.get_move()
        board.make_move(move)