import unittest
import numpy as np

from src.base import Game, Action, gameState, Piece, Role
from src.Games.connect4.connectFour import connect4_Board
from src.Games.hex.hex import hex_Board
from src.Games.TicTacToe.TicTacToe import TicTacToe_Game, TicTacToe_Action

class TestTicTacToe(unittest.TestCase):

    def test_init(self):
        game = TicTacToe_Game()
        xRole = Role('X')
        oRole = Role('O')

        self.assertTrue(np.all(game.board == None))
        self.assertEqual(len(game.legal_moves), 9)

        moves = []
        for i in range(9):
            moves.append(TicTacToe_Action(f"{i // 3}, {i % 3}", game.curr_role))
            self.assertTrue(moves[i] in game.legal_moves)

        self.assertEqual(game.all_actions, moves)
        self.assertEqual(game.roles, [xRole, oRole])
        self.assertEqual(game.roles[0].pieces, [])
        self.assertEqual(game.roles[1].pieces, [])

        self.assertEqual(game.state, gameState.ONGOING)
        self.assertEqual(game.winner, None)
        self.assertEqual(game.curr_role_idx, 0)
        self.assertEqual(game.curr_role, xRole)
        self.assertEqual(game.history, [])


    def test_makeMove(self):
        game = TicTacToe_Game()
        xRole = Role('X')
        oRole = Role('O')

        move = TicTacToe_Action("0,0")

        game.make_action(move)

        self.assertFalse(np.all(game.board == None))
        self.assertEqual(game.board[0, 0], Piece('X', game.roles[0], location=(0, 0)))

        self.assertEqual(len(game.legal_moves), 8)

        moves = []
        for i in range(1, 9):
            moves.append(TicTacToe_Action(f"{i % 3},{i // 3}"))
            self.assertTrue(moves[i - 1] in game.legal_moves)

        self.assertEqual(game.roles[0].pieces, [Piece('X', game.roles[0], location=(0, 0))])
        self.assertEqual(game.roles[1].pieces, [])

        self.assertEqual(game.state, gameState.ONGOING)
        self.assertEqual(game.winner, None)
        self.assertEqual(game.curr_role_idx, 1)
        self.assertEqual(game.curr_role, Role('O'))
        self.assertEqual(game.history, [move])

    def testUnmakeMove(self):
        game = TicTacToe_Game()
        xRole = Role('X')
        oRole = Role('O')

        move = TicTacToe_Action("0,0")

        game.make_action(move)

        game.unmake_action()

        self.assertTrue(np.all(game.board == None))

        self.assertEqual(len(game.legal_moves), 9)

        moves = []
        for i in range(9):
            moves.append(TicTacToe_Action(f"{i % 3},{i // 3}"))
            self.assertTrue(moves[i] in game.legal_moves)

        self.assertEqual(game.roles[0].pieces, [Piece('X', game.roles[0], location=(0, 0))])
        self.assertEqual(game.roles[1].pieces, [])

        self.assertEqual(game.state, gameState.ONGOING)
        self.assertEqual(game.winner, None)
        self.assertEqual(game.curr_role_idx, 0)
        self.assertEqual(game.curr_role, game.roles[0])
        self.assertEqual(game.history, [])
    
        