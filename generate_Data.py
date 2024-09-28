# from Models.MCTS_UCT import MCTS_uct_Node, MCTS_uct_Tree
# from Games.hex.hex import hex_Board, hex_Move
# from Games.connect4.connectFour import connect4_Board, connect4_Move
# from Games.Game import gameState
import random
import numpy as np
from src.base import gameState

from src.Games.connect4.connectFour import connect4_Board, connect4_Move
from src.players.MCTS.Models.MCTS_uct import MCTS_uct_Node, MCTS_uct_Tree


game_board = connect4_Board(6, 7, players=['X', 'O'])


while game_board.state == gameState.ONGOING:
    game_board.make_action(random.choice(game_board.legal_moves))

# game = MCTS_uct_Tree(game_board)
# game.move(random.choice(game.board.legal_moves))
# game.run([], engine_max_iter=1500, debug=True)
# print(game.board)

game_board = connect4_Board(6, 7, players=['X', 'O'])
game = MCTS_uct_Tree(game_board)
while game_board.state == gameState.ONGOING:
    game.search_tree(max_iter=1000)
    move, child = game.best()
    game.update_state(move)
print(game_board)
print("winner: ", game_board.winner) 

res = 0
if game_board.state == gameState.DRAW:
    res = 0.5
elif game_board.winner == game_board.players[0]:
    res = 1

samples = game_board.encode()
samples = samples.reshape((1, *samples.shape))

labels = np.array([res])
labels = labels.reshape((1, *labels.shape))
    
    
for i in range(499):
    game_board = connect4_Board(6, 7, players=['X', 'O'])
    game = MCTS_uct_Tree(game_board)
    while game_board.state == gameState.ONGOING:
        game.search_tree(max_iter=1000 + i*2)
        move, child = game.best()
        game.update_state(move)
    print(game_board)
    print("winner: ", game_board.winner)    
    
    new_sample = game_board.encode()
    samples = np.concatenate([samples, new_sample.reshape((1, *new_sample.shape))])
    res = 0
    if game_board.state == gameState.DRAW:
        res = 0.5
    elif game_board.winner == game_board.players[0]:
        res = 1
    new_label = np.array([res])
    labels = np.concatenate([labels, new_label.reshape((1, *new_label.shape))])

path = '/'.join(game_board.__module__.split(".")[0:2]) + '/Data/'

np.save(path + 'X_final_MCTS.npy', samples)
np.save(path + 'Y_final_MCTS.npy', labels)

# Load the saved NumPy arrays
X_train: np.ndarray = np.load(path + 'X_final_MCTS.npy')
y_train: np.ndarray = np.load(path + 'Y_final_MCTS.npy')

print(X_train)
print(np.count_nonzero(y_train))