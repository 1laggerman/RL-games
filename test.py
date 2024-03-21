from hex import hex_Board, hex_Move
from connectFour import connect4_Board, connect4_Move
from MCTS import MCTSTree, ALGORITHMS
import numpy as np
from collections import deque

# column_names = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

# def print_board(board):
#     rows = 5
#     cols = 5
#     indent = 0
#     headings = " "*5+(" "*3).join(column_names[:cols])
#     print(headings)
#     tops = " "*5+(" "*3).join("-"*cols)
#     print(tops)
#     roof = " "*4+"/ \\"+"_/ \\"*(cols-1)
#     print(roof)
#     color_mapping = lambda i : " WB"[i]
#     for r in range(rows):
#         row_mid = " "*indent
#         row_mid += " {} | ".format(r+1)
#         row_mid += " | ".join(map(color_mapping,board[r]))
#         row_mid += " | {} ".format(r+1)
#         print(row_mid)
#         row_bottom = " "*indent
#         row_bottom += " "*3+" \\_/"*cols
#         if r<rows-1:
#             row_bottom += " \\"
#         print(row_bottom)
#         indent += 2
#     headings = " "*(indent-2)+headings
#     print(headings)

# board=[[0,0,0,0],[0,0,0,1],[0,0,0,2],[1,2,0,0],[0,2,1,0]]
# print_board(board)



board = hex_Board(11, players=['X', 'O'])
board.make_move(board.create_move('1 1'))
board.print_board()

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
