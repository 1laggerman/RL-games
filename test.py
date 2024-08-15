from src.Games.TicTacToe.TicTacToe import TicTacToe_Board, TicTacToe_move
from src.players.Input.terminalInput import terminalPlayer as humanPlayer
from src.players.MCTS.Models.MCTS_uct import MCTS_uct_Tree
from src.players.MCTS.Models.Alpha_Zero import Alpha_Zero_player, AZ_search_args, AZ_NArgs, AZ_train_args
from src.players.MCTS.Models.ML_architecture.resnet import BaseRenset
from src.players.MCTS.Treeplayer import SArgs
from src.base import play, bind, Piece
import time
from copy import deepcopy
import numpy as np
from typing import Any
import torch
import os


board = TicTacToe_Board((3, 3))

# p1 = MCTS_uct_Tree(board, "X")
# p1 = humanPlayer(board, "X")

SearchArgs = AZ_search_args(max_iters=12) # TODO: check higher max_iters
net = BaseRenset(board, num_resblocks=4, num_hidden=64)
opt = torch.optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999), weight_decay=0.2)
value_crit = torch.nn.MSELoss()
policy_crit = torch.nn.CrossEntropyLoss()

NArgs = AZ_NArgs(net=net, optimizer=opt, value_criterion=value_crit, policy_criterion=policy_crit, device=net.device)
p1 = Alpha_Zero_player(board, "X", SearchArgs, net_args=NArgs)


p2 = humanPlayer(board, "O")

players = [p1, p2]


bind(board, players)

p1.load_model('base')

args = AZ_train_args(batch_epochs=5, batch_size=500, num_batches=10, reward_decay=0.9)
p1.train(args)

p1.save_model('test', override=False)





# A = p1.self_play(decay=0.9)
# A = p1.self_play_batch(10, decay=0.9)

# print(A)
# # print(samples.shape, value_labels.shape, policy_labels.shape)

# print(samples, '\n', value_labels, '\n', policy_labels)

