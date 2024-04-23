from Games.hex.hex import hex_Board, hex_Move
from Games.connect4.connectFour import connect4_Board, connect4_Move
from Games.Game import gameState
from Models.MCTS_UCT import MCTS_uct_Tree
# from MCTS import MCTSTree, ALGORITHMS
from Models.MCTS_NN import MCTS_NN_Node, MCTS_NN_Tree
import numpy as np
from collections import deque

board = hex_Board(5, players=['X', 'O'])
# print(board.encode())
game = MCTS_NN_Tree(board)

X_train = np.load('Games/hex/Data/X_final_MCTS.npy')
Y_train = np.load('Games/hex/Data/Y_final_MCTS.npy')
game.static_train(70, X_train, Y_train, save_to="Games/hex/Models", save_as=f"net_1")

X_test = np.load('Games/hex/Data/X_final_random.npy')
Y_test = np.load('Games/hex/Data/Y_final_random.npy')

game.static_test(X_train, Y_train)
game.static_test(X_test, Y_test)


# game.train(self_learn_epochs=25, game_epochs=1, num_searches=100, new=True)
# game.run(['X'], engine_max_iter=100)

# game.static_test(X_train, Y_train)
# game.static_test(X_test, Y_test)

# game = MCTSTree(board)
# game.run(['X'], engine_max_iter=1000, debug=True)
# board.make_move(board.create_move("0 0"))
# board.make_move(board.create_move("1 1"))
# board.make_move(board.create_move("1 0"))
# board.make_move(board.create_move("2 2"))
# board.make_move(board.create_move("2 0"))
# board.make_move(board.create_move("3 3"))
# board.make_move(board.create_move("3 0"))
# board.make_move(board.create_move("4 4"))
# board.make_move(board.create_move("4 0"))
# print(board)
# game = MCTS_uct_Tree(board)
# game.run(['X'], engine_max_iter=1000, debug=True)


# a = np.array([[1, 2, 3]])
# b = np.array([[4, 5, 6]])
# a = np.array([[[1, 2, 3], [1, 1, 1]], [[0, -1, -3], [0, 0, 0]]])
# b = np.array([[4, 5, 6], [7, 8, 9]])
# print(a)
# # print(b)
# b = b.reshape((1, *b.shape))
# print(b)

# t = np.concatenate([a, b])

# print(t)