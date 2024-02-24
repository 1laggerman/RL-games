from connectFour import connect4_Move, connect4_Board, gameState

game = connect4_Board(rows=6, cols=7)
# while game.state == gameState.ONGOING:
#     print(game)
#     m = int(input(f"{game.curr_player}'s move: \nlegal moves(column number): {game.legal_moves}\nEnter your move: "))
#     move = connect4_Move(str(m), game.curr_player, m)
#     if game.is_legal_move(move):
#         game.make_move(move)
        

# print(game)
# print('Game over!')
# if game.state == gameState.DRAW:
#     print('The game ended in a draw!')
# else:
#     print(f'{game.winner} WON!')

from MCTSNode import MCTSNode

node = MCTSNode()

node.simulate(game)

print(game)
print(node.visits)
print(node.wins)

