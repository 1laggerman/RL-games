from src.Games.TicTacToe.TicTacToe import TicTacToe_Board, TicTacToe_move
from src.players.Input.terminalInput import terminalPlayer as humanPlayer
from src.players.MCTS.Models.MCTS_uct import MCTS_uct_Tree
from src.players.MCTS.Models.Alpha_Zero import Alpha_Zero_player, AZ_search_args, AZ_net_args
from src.players.MCTS.Models.ML_architecture.resnet import BaseRenset
from src.players.MCTS.Treeplayer import searchArgs
from src.base import play, bind, Piece
import time
from copy import deepcopy
import numpy as np
from typing import Any
import torch

# move = TicTacToe_move("0, 0")
# print(move)
# board = TicTacToe_Board((3, 3), players=[player(None, "O"), player(None, "X")])

board = TicTacToe_Board((3, 3))

# p1 = MCTS_uct_Tree(board, "X")
# p1 = humanPlayer(board, "X")

SArgs = AZ_search_args(max_iters=10)
net = BaseRenset(board, num_resblocks=10, num_hidden=3)
opt = torch.optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999), weight_decay=0.4)
value_crit = torch.nn.MSELoss()
policy_crit = torch.nn.CrossEntropyLoss()
NArgs = AZ_net_args(net=net, optimizer=opt, value_criterion=value_crit, policy_criterion=policy_crit, device=net.device)
p1 = Alpha_Zero_player(board, "X", SArgs, net_args=NArgs)


p2 = humanPlayer(board, "O")

players = [p1, p2]


bind(board, players)

# print(board.map_move(TicTacToe_move("1, 1")))

# board.map_move()

p1.self_play(decay=0.9)
# board.make_move(TicTacToe_move("0, 0"))
# print(board.encode())


# play(board, players=players)

# start = time.time()

# b2 = deepcopy(board)

# end = time.time()

# print(end - start)

# n = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# print(n[(1, 1)])


