from Models.MCTS_UCT import MCTS_uct_Node, MCTS_uct_Tree
from Games.hex.hex import hex_Board, hex_Move, gameState
import random
import numpy as np

# game_board = hex_Board(5)
# while game_board.state == gameState.ONGOING:
#     game_board.make_move(random.choice(game_board.legal_moves))
# # game = MCTS_uct_Tree(game_board)
# # game.move(random.choice(game.board.legal_moves))
# # game.run([], engine_max_iter=1500, debug=True)
# # print(game.board)

# res = 0
# if game_board.state == gameState.DRAW:
#     res = 0.5
# elif game_board.winner == game_board.players[0]:
#     res = 1

# samples = game_board.encode()
# samples = samples.reshape((1, *samples.shape))

# labels = np.array([res])
# labels = labels.reshape((1, *labels.shape))
    
    
# for i in range(499):
#     game_board = hex_Board(5)
#     while game_board.state == gameState.ONGOING:
#         game_board.make_move(random.choice(game_board.legal_moves))
#     print(game_board)
#     print("winner: ", game_board.winner)    
    
#     new_sample = game_board.encode()
#     samples = np.concatenate([samples, new_sample.reshape((1, *new_sample.shape))])
#     res = 0
#     if game_board.state == gameState.DRAW:
#         res = 0.5
#     elif game_board.winner == game_board.players[0]:
#         res = 1
#     new_label = np.array([res])
#     labels = np.concatenate([labels, new_label.reshape((1, *new_label.shape))])

# np.save('Data/X_test.npy', samples)
# np.save('Data/Y_test.npy', labels)

# Load the saved NumPy arrays
X_train: np.ndarray = np.load('Data/X_test.npy')
y_train: np.ndarray = np.load('Data/y_test.npy')

print(X_train)
print(y_train)