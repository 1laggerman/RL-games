from new_package.Games.TicTacToe.TicTacToe import TicTacToe_Board, TicTacToe_move
from new_package.players.Input.terminalInput import terminalPlayer as humanPlayer
from new_package.players.MCTS.Models.MCTS_uct import MCTS_uct_Tree
from new_package.base import play
# from package.gameplay.play import play


# move = TicTacToe_move("0, 0")
# print(move)
# board = TicTacToe_Board((3, 3), players=[player(None, "O"), player(None, "X")])

# p1 = humanPlayer(None, "X")
# players = [p1, p2]
board = TicTacToe_Board((3, 3))
p1 = MCTS_uct_Tree(board, "X")
p2 = humanPlayer(board, "O")
players = [p1, p2]

# print(board)
play(board=board, players=players)


