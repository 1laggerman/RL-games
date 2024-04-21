from MCTS_UCT import MCTS_uct_Node, MCTS_uct_Tree
from hex import hex_Board, hex_Move, gameState
import random
import numpy as np

# game_board = hex_Board(5)
# game = MCTS_uct_Tree(game_board)
# game.move(random.choice(game.board.legal_moves))
# game.run([], engine_max_iter=1500, debug=True)
# print(game.board)

# res = 0
# if game.board.state == gameState.DRAW:
#     res = 0.5
# elif game.board.winner == game.board.players[0]:
#     res = 1

# samples = game.board.encode()
# samples = samples.reshape((1, *samples.shape))

# labels = np.array([res])
# labels = labels.reshape((1, *labels.shape))
    
    
# for i in range(499):
#     game_board = hex_Board(5)
#     game = MCTS_uct_Tree(game_board)
#     game.move(random.choice(game.board.legal_moves))
#     game.run([], engine_max_iter=1500 + 2 * i)
#     new_sample = game.board.encode()
#     samples = np.concatenate([samples, new_sample.reshape((1, *new_sample.shape))])
#     res = 0
#     if game.board.state == gameState.DRAW:
#         res = 0.5
#     elif game.board.winner == game.board.players[0]:
#         res = 1
#     new_label = np.array([res])
#     labels = np.concatenate([labels, new_label.reshape((1, *new_label.shape))])

# Load the saved NumPy arrays
# X_train = np.load('Data/X_train.npy')
y_train: np.ndarray = np.load('Data/y_train.npy')

# print(X_train)
print(np.count_nonzero(y_train))