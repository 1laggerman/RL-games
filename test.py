from Games.TicTacToe.TicTacToe import TicTacToe_Board, TicTacToe_Move
from Games.connect4.connectFour import connect4_Board, connect4_Move
from Games.Game import gameState
from Models.MCTS_UCT import MCTS_uct_Tree
# from MCTS import MCTSTree, ALGORITHMS
from Models.MCTS_NN import MCTS_NN_Node, MCTS_NN_Tree
import numpy as np
from collections import deque
from copy import deepcopy


board = TicTacToe_Board((3, 3), ['X', 'O'])

