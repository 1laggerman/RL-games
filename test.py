from new_package.Games.TicTacToe.TicTacToe import TicTacToe_Board
from new_package.players.Input.terminalInput import terminalPlayer as humanPlayer
from new_package.base import play
# from package.gameplay.play import play


# board = TicTacToe_Board((3, 3), players=[player(None, "O"), player(None, "X")])

p1 = humanPlayer(None, "O")
p2 = humanPlayer(None, "X")
players = [p1, p2]
board = TicTacToe_Board((3, 3))

play(board=board, players=players)


