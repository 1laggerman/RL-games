from Game import Move, Board, gameState
from Tree import Node, SearchTree
from hex import hex_Move
import math
from copy import deepcopy
import numpy as np

import torch
import matplotlib.pyplot as plt

torch.manual_seed(0)

class value_head(torch.nn.Module):
    def __init__(self, board: Board, num_resblocks: int, num_hidden: int) -> None:
        super().__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        self.start_block = torch.nn.Sequential(
            torch.nn.Conv2d(3, num_hidden, kernel_size=3, padding=1, device=self.device),
            torch.nn.BatchNorm2d(num_hidden, device=self.device),
            torch.nn.Conv2d(3, num_hidden, kernel_size=3, padding=1, device=self.device),
            torch.nn.ReLU()
        )
        
        self.backBone = torch.nn.ModuleList([ResBlock(num_hidden) for _ in range(num_resblocks)])
        
        self.policy_head = torch.nn.Sequential(
            torch.nn.Conv2d(num_hidden, 32, kernel_size=3, padding=1, device=self.device),
            torch.nn.BatchNorm2d(32, device=self.device),
            torch.nn.ReLU(),
            torch.nn.Flatten(),
            torch.nn.Linear(32 * board.board.shape[0] * board.board.shape[1], len(board.legal_moves), device=self.device), # test this
        )
        
        self.value_head = torch.nn.Sequential(
            torch.nn.Conv2d(num_hidden, 3, kernel_size=3, padding=1, device=self.device),
            torch.nn.BatchNorm2d(3, device=self.device),
            torch.nn.ReLU(),
            torch.nn.Flatten(),
            torch.nn.Linear(3 * board.board.shape[0] * board.board.shape[1], 1, device=self.device),
            torch.nn.Sigmoid()
        )
        
        # shape = board.encode().shape
        # in_features = 1
        # for size in shape:
        #     in_features *= size 
        # self.fc1 = torch.nn.Linear(in_features, 32, device=self.device)
        # self.fc2 = torch.nn.Linear(32, 1, device=self.device)
        
    # def forward(self, x: torch.Tensor):
    #     x = x.view((x.shape[0], -1))
    #     x = torch.nn.functional.relu(self.fc1(x))
    #     x = torch.nn.functional.sigmoid(self.fc2(x))
    #     return x
    
    def forward(self, x: torch.Tensor):
        x = self.start_block(x)
        for resBlock in self.backBone:
            x = resBlock(x)
        # policy = self.policy_head(x)
        value = self.value_head(x)
        return value


class ResBlock(torch.nn.Module):
    def __init__(self, num_hidden: int) -> None:
        super().__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.conv1 = torch.nn.Conv2d(num_hidden, num_hidden, kernel_size=3, padding=1, device=self.device)
        self.bn1 = torch.nn.BatchNorm2d(num_hidden, device=self.device)
        self.conv2 = torch.nn.Conv2d(num_hidden, num_hidden, kernel_size=3, padding=1, device=self.device)
        self.bn2 = torch.nn.BatchNorm2d(num_hidden, device=self.device)
        
    def forward(self, x: torch.Tensor):
        residual = x
        x = torch.nn.functional.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual
        x = torch.nn.functional.relu(x)
        return x

class MCTS_NN_Node(Node):
    
    def __init__(self, board: Board, vh: torch.nn.Module, parent: "MCTS_NN_Node" = None) -> None:
        super(MCTS_NN_Node, self).__init__(untried_actions=board.legal_moves, player=board.curr_player, parent=parent)
        self.vh = vh
        self.tree_eval = 0
        self.final_eval = 0
        
    def select_child(self):
        
        assert self.visits > 0, "Parent node has zero visits."
        # Exploration parameter
        C = math.sqrt(2)

        # Calculate UCT score for each child and select the child with the highest score
        best_score = float('-inf')
        best_child = None
        for child in self.children:
            # exploiting 1 - (wins / visits) because the child node is a different player
            exploitation = (child[1].visits - child[1].eval) / child[1].visits if child[1].visits > 0 else 0
            exploration = math.sqrt(math.log(self.visits) / (1 + child[1].visits))
            uct_score = exploitation + C * exploration

            if uct_score > best_score:
                best_score = uct_score
                best_child = child

        return best_child
         
    def expand(self, board: Board, move: Move = None):
        new_action = move
        if move is None or move not in self.untried_actions:
            new_action = self.untried_actions.pop()
        
        board.make_move(new_action)
        new_Node = MCTS_NN_Node(board, vh=self.vh, parent=self)
        if board.state == gameState.ENDED or board.state == gameState.DRAW:
            new_Node.is_leaf = True
        board.unmake_move()
        new_child = (new_action, new_Node)
        self.children.append(new_child)
        return new_child
        
    def evaluate(self, board: Board):
        return self.vh.forward(torch.Tensor(board.encode()).unsqueeze(0).to(self.vh.device))
    
    def backpropagate(self, eval: float):
        self.visits += 1
        self.eval = eval
        if self.parent:
            self.parent.bp(1-eval)
            
    def bp(self, eval: float):
        self.tree_eval += eval
        self.visits += 1
        if self.parent:
            self.parent.bp(1-eval)
    
class MCTS_NN_Tree(SearchTree):
    
    def __init__(self, game_board: Board) -> None:
        super(MCTS_NN_Tree, self).__init__(game_board)
        self.vh = value_head(game_board, num_resblocks=10, num_hidden=3)
        # self.vh = torch.load('value_head.pt')
        self.crit = torch.nn.MSELoss()
        self.opt = torch.optim.Adam(self.vh.parameters(), lr=0.1, betas=(0.9, 0.999), weight_decay=0.1)
        with torch.no_grad():
            self.vh.apply(init_weights)
        self.root = MCTS_NN_Node(board=game_board, vh=self.vh)
        self.root.eval = self.root.evaluate(game_board)
        self.root.visits = 1
        
    def best(self):
        return min(self.root.children, key=lambda c: c[1].eval + c[1].tree_eval / (2 * c[1].visits) if c[1].visits > 0 else 0)
    
    def train(self, self_learn_epochs: int, game_epochs: int, num_searches: int = 1000, tree_max_depth: int = -1):
        board_backup = deepcopy(self.board)
        self.vh = torch.load('value_head.pt')
        losses = []
        
        for i in range(self_learn_epochs):
            self.board = deepcopy(board_backup)
            # print(self.board)
            while self.board.state == gameState.ONGOING:
                self.calc_best_move(max_iter=num_searches, max_depth=tree_max_depth)
                move, child = self.best()
                
                if move is not None:
                    self.move(move)
                else:
                    print("ERROR")
                    return
                # print(self.board)
            print(self.board)
            print('winner is: ', self.board.winner)
            res = 0
            if self.board.state == gameState.DRAW:
                res = 0.5
            elif self.board.winner == self.board.curr_player:
                res = 1
            
            # pred = child.eval
            samples = self.board.encode()
            samples = samples.reshape((1, *samples.shape))
            # labels = torch.Tensor([res]).to(self.vh.device)
            labels = np.array([res])
            labels = labels.reshape((1, *labels.shape))
            res = 1 - res
            parent = child.parent
            while parent is not None:
                self.board.unmake_move()
                
                new_sample = self.board.encode()
                samples = np.concatenate([samples, new_sample.reshape((1, *new_sample.shape))])
                torch.FloatTensor()
                new_label = np.array([res])
                labels = np.concatenate([labels, new_label.reshape((1, *new_label.shape))])
                # pred = torch.cat((pred, parent.eval), dim=0)
                # labels = torch.cat((labels, torch.Tensor([res]).to(self.vh.device)), dim=0)
                
                res = 1 - res
                parent = parent.parent
            
            samples = torch.tensor(samples).to(self.vh.device)
            labels = torch.tensor(labels.astype(np.float32)).to(self.vh.device)
            
            for epoch in range(game_epochs):
                self.opt.zero_grad()
                pred = self.vh.forward(samples)
                loss = self.crit.forward(pred, labels)
                losses.append(loss.item())
                loss.backward()
                self.opt.step()
                
            self.root = MCTS_NN_Node(board=self.board, vh=self.vh)
            self.root.eval = self.root.evaluate(self.board)
            self.root.visits = 1
        
        plt.plot(losses)
        plt.show()
        self.board = deepcopy(board_backup)
        torch.save(self.vh, "value_head.pt")
                
            
            
                 
def init_weights(m):
    if type(m) == torch.nn.Linear:
        m.weight.fill_(0.001)
        m.bias.fill_(0.001)