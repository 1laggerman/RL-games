from hex import hex_Board, hex_Move
from connectFour import connect4_Board, connect4_Move
from Game import gameState
from MCTS import MCTSTree, ALGORITHMS
import numpy as np
from collections import deque


board = hex_Board(5, players=['X', 'O'])
game = MCTSTree(board)
game.run(['X'])

# i = 0
# while board.state == gameState.ONGOING and i < 5:
#     print(board)
#     print(board.legal_moves)
#     m = input(f"enter move: ")
#     move = board.create_move(m)
#     board.make_move(move)
#     print(board.linked_to_edge)
#     i += 1


# while(i > 0):
#     board.unmake_move()
#     print(board)
#     print(board.legal_moves)
#     input("next")
#     i -= 1

# board = connect4_Board(6, 7)
# game = MCTSTree(board)
# game.run(input_players=['X'], debug=True, engine_max_iter=1000, engine_max_depth=-1)

# print(hex_Move("0 0", (0, 0)) == hex_Move("0 0", (0, 0)))

# a = np.zeros((2, 2, 5, 5), dtype=bool)
# print(a)

# n = np.array([[[True, False], [False, True]], [[True, False], [False, True]]])

# found_edges = n[:, *(0, 1)]
# for i, edge in enumerate(found_edges):
#     if edge == True:
#         print(i)

# def foo(a: int):
#     print(a)
#     print(type(a))

# b = 0
# foo(0 if b == 1 else 1)
# q = deque([3])
# print(q.pop())

# n = np.array([[[0, 1], [2, 3]], [[4, 5], [6, 7]]])
# location = [0, 0]
# print(n)
# print(n[:, *location])
# print(n[(1, 0)])

# b = hex_Board(11, players=['X', 'O'])
# print(b.get_links(move=hex_Move("0 0", (0, 0))))
