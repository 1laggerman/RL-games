from hex import hex_Board, hex_Move
from MCTS import MCTSTree, ALGORITHMS
import numpy as np
from collections import deque

# board = connect4_Board(6, 7)
# game = MCTSTree(board)
# game.run(input_players=['X'], debug=True, engine_max_iter=1000, engine_max_depth=-1)

# print(hex_Move("0 0", (0, 0)) == hex_Move("0 0", (0, 0)))

# a = np.zeros((2, 2, 5, 5), dtype=bool)
# print(a)

def foo(a: int):
    print(a)
    print(type(a))

b = 0
foo(0 if b == 1 else 1)
# q = deque([3])
# print(q.pop())

# n = np.array([[[0, 1], [2, 3]], [[4, 5], [6, 7]]])
# location = [0, 0]
# print(n)
# print(n[:, *location])
# print(n[(1, 0)])

# b = hex_Board(11, players=['X', 'O'])
# print(b.get_links(move=hex_Move("0 0", (0, 0))))
