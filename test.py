from connectFour import connect4_Board
from MCTS import MCTSTree, ALGORITHMS

board = connect4_Board(6, 7)
game = MCTSTree(board)
game.run(input_players=['X'], debug=True, alg=ALGORITHMS.EPSILON_GREEDY, epsilon=0.2, engine_max_iter=1000, engine_max_depth=-1)
