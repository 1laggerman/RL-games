from MCTS_UCT import MCTS_uct_Node, MCTS_uct_Tree
from hex import hex_Board, hex_Move, gameState
import random
import numpy as np

game_board = hex_Board(5)
game = MCTS_uct_Tree(game_board)
game.move(random.choice(game.board.legal_moves))
game.run([], engine_max_iter=3000, debug=True)
print(game.board)

res = 0
if game.board.state == gameState.DRAW:
    res = 0.5
elif game.board.winner == game.board.players[0]:
    res = 1

samples = game.board.encode()
samples = samples.reshape((1, *samples.shape))

labels = np.array([res])
labels = labels.reshape((1, *labels.shape))
    
    
for i in range(2):
    game_board = hex_Board(5)
    game = MCTS_uct_Tree(game_board)
    game.move(random.choice(game.board.legal_moves))
    game.run([], engine_max_iter=3000)
    new_sample = game.board.encode()
    samples = np.concatenate([samples, new_sample.reshape((1, *new_sample.shape))])
    new_label = np.array([res])
    labels = np.concatenate([labels, new_label.reshape((1, *new_label.shape))])

np.save('Data/X_train.npy', samples)
np.save('Data/y_train.npy', labels) 

# Load the saved NumPy arrays
X_train = np.load('Data/X_train.npy')
y_train = np.load('Data/y_train.npy')

print(X_train)
print(y_train)