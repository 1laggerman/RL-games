from hex import hex_Board, hex_Move
from MCTS import MCTSTree, ALGORITHMS

# board = connect4_Board(6, 7)
# game = MCTSTree(board)
# game.run(input_players=['X'], debug=True, engine_max_iter=1000, engine_max_depth=-1)

print(hex_Move("0 0", (0, 0)) == hex_Move("0 0", (0, 0)))

# b = hex_Board(11, players=['X', 'O'])
# print(b.get_links(move=hex_Move("0 0", (0, 0))))
