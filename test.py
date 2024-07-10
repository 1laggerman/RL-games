from src.Games.TicTacToe.TicTacToe import TicTacToe_Board, TicTacToe_move
from src.players.Input.terminalInput import terminalPlayer as humanPlayer
from src.players.MCTS.Models.MCTS_uct import MCTS_uct_Tree
from src.players.MCTS.Models.Alpha_Zero import Alpha_Zero_player
from src.players.MCTS.Models.ML_architecture.resnet import BaseRenset
from src.base import play, bind, Piece
import time
from copy import deepcopy
import numpy as np
from typing import Any
# from package.gameplay.play import play

# move = TicTacToe_move("0, 0")
# print(move)
# board = TicTacToe_Board((3, 3), players=[player(None, "O"), player(None, "X")])

board = TicTacToe_Board((3, 3))

# p1 = MCTS_uct_Tree(board, "X")
p1 = Alpha_Zero_player(board, "X", net=BaseRenset(board, 10, 3))
p2 = humanPlayer(board, "O")

players = [p1, p2]


bind(board, players)

# print(board.map_move(TicTacToe_move("1, 1")))

# board.map_move()

p1.self_play(decay=0.9)
# board.make_move(TicTacToe_move("0, 0"))
# print(board.encode())


# play(board=board, players=players)

# start = time.time()

# b2 = deepcopy(board)

# end = time.time()

# print(end - start)

# n = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# print(n[(1, 1)])


