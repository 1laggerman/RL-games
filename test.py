from new_package.Games.TicTacToe.TicTacToe import TicTacToe_Board, TicTacToe_move
from new_package.players.Input.terminalInput import terminalPlayer as humanPlayer
from new_package.base import play
# from package.gameplay.play import play

# move = TicTacToe_move("0, 0")
# print(move)
# board = TicTacToe_Board((3, 3), players=[player(None, "O"), player(None, "X")])

p1 = humanPlayer(None, "X")
p2 = humanPlayer(None, "O")
players = [p1, p2]
board = TicTacToe_Board((3, 3), players=players)

# print(board)
play(board=board, players=players)


