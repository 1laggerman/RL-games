from src.Games.TicTacToe.TicTacToe import TicTacToe_Game, TicTacToe_Action
from src.players.Input.terminalInput import terminalPlayer as humanPlayer
from src.players.MCTS.Models.MCTS_uct import MCTS_uct_Tree
from src.players.MCTS.Models.Alpha_Zero import Alpha_Zero_player, AZ_search_args, AZ_NArgs, AZ_train_args
from src.players.MCTS.Models.ML_architecture.resnet import BaseResnet
from src.players.MCTS.Treeplayer import SArgs
from src.base import play, bind, Piece
import time
from copy import deepcopy
import numpy as np
from typing import Any
import torch
import os



# game = TicTacToe_Game()

# print(game.roles)
# xrole = game.roles[0]
# orole = game.roles[1]


# print(xrole.pieces)
# print(orole.pieces)

# game.make_move(TicTacToe_move('0, 0'))


# print(xrole.pieces)
# print(orole.pieces)


# p1 = MCTS_uct_Tree(board, "X")
# p1 = humanPlayer(board, "X")


# board = TicTacToe_Board((3, 3))

# SearchArgs = AZ_search_args(max_iters=60)
# net = BaseResnet(board, num_resblocks=4, num_hidden=64)
# opt = torch.optim.Adam(net.parameters(), lr=0.01)
# value_crit = torch.nn.MSELoss()
# policy_crit = torch.nn.CrossEntropyLoss()

# NArgs = AZ_NArgs(net=net, optimizer=opt, value_criterion=value_crit, policy_criterion=policy_crit, device=net.device)
# p1 = Alpha_Zero_player(board, "X", SearchArgs, net_args=NArgs)

# p2 = humanPlayer(board, "O")

# players = [p1, p2]


# bind(board, players)

# res = p1.self_play(decay=0.9)
# print(res.X)

# args = AZ_train_args(batch_epochs=4, batch_size=500, num_batches=3, reward_decay=0.9)
# p1.train(args)

# p1.save_model('base', override=True)

# play(board, players)








from src.players.MCTS.Models.Alphatest import AlphaZero, TicTacToe, ResNet
from src.players.MCTS.Models.ML_architecture.resnet import BaseResnet
import matplotlib.pyplot as plt
import torch

tictactoe = TicTacToe()
# board = TicTacToe_Board((3, 3))

model = ResNet(4, 64)
# model = BaseResnet(4, 64)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

args = {
    'C': 2,
    'num_searches': 60,
    'num_iterations': 3,
    'num_selfPlay_iterations': 500,
    'num_epochs': 20,
    'batch_size': 64
}

alphaZero = AlphaZero(model, optimizer, tictactoe, args)
alphaZero.learn()
# alphaZero.learn()

tictactoe = TicTacToe()

state = tictactoe.get_initial_state()
state = tictactoe.get_next_state(state, 2, -1)
state = tictactoe.get_next_state(state, 4, -1)
state = tictactoe.get_next_state(state, 6, 1)
state = tictactoe.get_next_state(state, 8, 1)


encoded_state = tictactoe.get_encoded_state(state)

tensor_state = torch.tensor(encoded_state).unsqueeze(0)

# model = ResNet(tictactoe, 4, 64)
# model.load_state_dict(torch.load('model_2.pt'))
model.eval()

policy, value = model(tensor_state)
value = value.item()

if type(model) == ResNet:
    policy = torch.softmax(policy, axis=1).squeeze(0).detach().cpu().numpy()
else:
    policy = policy.squeeze(0).detach().cpu().numpy()
# policy = policy.squeeze(0).detach().cpu().numpy()
# policy = torch.softmax(policy, axis=1).squeeze(0).detach().cpu().numpy()

print(policy.shape)

print(value)

print(state)
print(tensor_state)

plt.bar(range(tictactoe.action_size), policy)
# plt.savefig('')
plt.show()



# mem = alphaZero.selfPlay()


# def to_tensor(memory):
#     state, policy_targets, value_targets = zip(*memory)
            
#     state, policy_targets, value_targets = np.array(state), np.array(policy_targets), np.array(value_targets).reshape(-1, 1)
    
#     state = torch.tensor(state, dtype=torch.float32)
#     policy_targets = torch.tensor(policy_targets, dtype=torch.float32)
#     value_targets = torch.tensor(value_targets, dtype=torch.float32)

#     print(state[0])

#     for i in range(len(memory)):
#         memory[i] = torch.tensor(memory[i])
#     return memory

# # print(res[0])

# # print(state[0])
# tens = to_tensor(mem)
# # print(tens)
# # print(to_tensor(res))
