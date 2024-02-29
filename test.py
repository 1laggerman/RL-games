from connectFour import connect4_Move, connect4_Board, gameState
from MCTS import MCTSTree

board = connect4_Board(6, 7)
game = MCTSTree(board)
game.run(input_players=['X'], debug=True, engine_max_iter=1000, engine_max_depth=-1)
