from src.Games.TicTacToe.TicTacToe import TicTacToe_Board, TicTacToe_move
from src.players.Input.terminalInput import terminalPlayer as humanPlayer
from src.players.MCTS.Models.MCTS_uct import MCTS_uct_Tree
from src.players.MCTS.Models.Alpha_Zero import Alpha_Zero_player
from src.players.MCTS.Models.ML_architecture.resnet import BaseRenset
from src.players.MCTS.Treeplayer import searchArgs
from src.base import play, bind, Piece
import time
from copy import deepcopy
import numpy as np
from typing import Any
from dataclasses import dataclass

# @dataclass
# class B(object):
#     __slots__ = ('x')
#     def __init__(self, x=7):
#         self.x = x

# b = B()

# print(b.x)

# searchArgs.max_iters = 15
# searchArgs.max_depth = 3
# searchArgs.max_time = 5
args = searchArgs(max_iters=10)

print(args.max_depth, args.max_iters, args.max_time)

# move = TicTacToe_move("0, 0")
# print(move)
# board = TicTacToe_Board((3, 3), players=[player(None, "O"), player(None, "X")])

# board = TicTacToe_Board((3, 3))

# p1 = MCTS_uct_Tree(board, "X")
# p1 = Alpha_Zero_player(board, "X", net=BaseRenset(board, 10, 3))
# p1 = humanPlayer(board, "X")

# p2 = humanPlayer(board, "O")

# players = [p1, p2]


# bind(board, players)

# print(board.map_move(TicTacToe_move("1, 1")))

# board.map_move()

# p1.self_play(decay=0.9)
# board.make_move(TicTacToe_move("0, 0"))
# print(board.encode())


# play(board, players=players)

# start = time.time()

# b2 = deepcopy(board)

# end = time.time()

# print(end - start)

# n = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# print(n[(1, 1)])


