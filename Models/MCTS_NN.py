from Games.Game import Move, Board, gameState
from Models.Tree import Node, SearchTree
import math
import random
from copy import deepcopy
import numpy as np

import torch
import matplotlib.pyplot as plt
from torchmetrics.classification import confusion_matrix, accuracy

torch.manual_seed(0)


class resnet(torch.nn.Module):
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
            torch.nn.Softmax(dim=1)
        )
        
        self.value_head = torch.nn.Sequential(
            torch.nn.Conv2d(num_hidden, 3, kernel_size=3, padding=1, device=self.device),
            torch.nn.BatchNorm2d(3, device=self.device),
            torch.nn.ReLU(),
            torch.nn.Flatten(),
            torch.nn.Linear(3 * board.board.shape[0] * board.board.shape[1], 1, device=self.device),
        )
    
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.start_block(x)
        for resBlock in self.backBone:
            x = resBlock(x)
        policy = self.policy_head(x)
        value = self.value_head(x)
        return value, policy


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
    policy: np.ndarray
    prob: float
    
    def __init__(self, board: Board, net: torch.nn.Module, parent: "MCTS_NN_Node" = None, prob: float = 0) -> None:
        super(MCTS_NN_Node, self).__init__(untried_actions=board.legal_moves, player=board.curr_player, parent=parent)
        self.net: resnet = net
        self.tree_eval = 0
        self.final_eval = 0
        self.prob = prob
        with torch.no_grad():
            self.eval, self.policy = self.evaluate(board)
        
      
    def select_child(self):
        
        # Exploration parameter
        C = math.sqrt(2)

        # Calculate UCT score for each child and select the child with the highest score
        best_score = float('-inf')
        best_child = None
        
        exploitation = math.sqrt(self.visits)
        for child in self.children:
            exploitation = 1 - (child[1].eval / (child[1].visits + 1)) if child[1].visits > 0 else 0
            exploration = math.sqrt(self.visits) / (1 + child[1].visits)
            uct_score = exploitation + C * exploration * child[1].prob

            if uct_score > best_score:
                best_score = uct_score
                best_child = child

        return best_child  
        
    # def select_child(self):
        
    #     assert self.visits > 0, "Parent node has zero visits."
    #     # Exploration parameter
    #     C = math.sqrt(2)

    #     # Calculate UCT score for each child and select the child with the highest score
    #     best_score = float('-inf')
    #     best_child = None
    #     for child in self.children:
    #         exploitation = 1 - child[1].eval
    #         exploration = math.sqrt(math.log(self.visits) / (1 + child[1].visits))
    #         uct_score = exploitation + C * exploration

    #         if uct_score > best_score:
    #             best_score = uct_score
    #             best_child = child

    #     return best_child
    
    
    def expand(self, board: Board) -> None:
        for prob in self.policy:
            if prob > 0:
                new_action = self.untried_actions.pop()
                board.make_move(new_action)
                new_Node = MCTS_NN_Node(board, net=self.net, parent=self, prob=prob)
                if board.state == gameState.ENDED or board.state == gameState.DRAW:
                    new_Node.is_leaf = True
                # new_Node.eval, new_Node.policy = self.evaluate(board)
                board.unmake_move()
                new_child = (new_action, new_Node)
                self.children.append(new_child)
        
        return random.choices(self.children, weights=self.policy[self.policy > 0], k=1)[0]
         
    # def expand(self, board: Board, move: Move = None):
    #     new_action = move
    #     if move is None or move not in self.untried_actions:
    #         new_action = self.untried_actions.pop()
        
    #     board.make_move(new_action)
    #     new_Node = MCTS_NN_Node(board, vh=self.net, parent=self)
    #     if board.state == gameState.ENDED or board.state == gameState.DRAW:
    #         new_Node.is_leaf = True
    #     new_Node.eval = self.evaluate(board)
    #     board.unmake_move()
    #     new_child = (new_action, new_Node)
    #     self.children.append(new_child)
    #     return new_child
        
    def evaluate(self, board: Board) -> tuple[torch.Tensor, torch.Tensor]:
        value, policy = self.net.forward(torch.Tensor(board.encode()).unsqueeze(0).to(self.net.device))
        if self.player == board.players[1]:
            value = 1 - value
        
        # print(value.squeeze(0).detach().cpu().numpy()[0])
        policy = policy.squeeze(0).detach().cpu().numpy()
        legal = np.where(board.board == ' ', 1, 0).flatten()
        policy *= legal
        policy /= np.sum(policy)
        return value, policy
    
    def backpropagate(self, eval: float):
        self.visits += 1
        
        # best eval:
        # if eval > self.eval:
        #     self.eval = eval
        
        # average eval:
        self.eval += eval 
        
        if self.parent:
            self.parent.backpropagate(1-eval)
            
    # def bp(self, eval: float):
    #     if eval > self.eval:
    #         self.eval = eval
    #     self.visits += 1
    #     if self.parent:
    #         self.parent.bp(1-eval)
    
class MCTS_NN_Tree(SearchTree):
    
    def __init__(self, game_board: Board) -> None:
        super(MCTS_NN_Tree, self).__init__(game_board)
        self.net: resnet = resnet(game_board, num_resblocks=10, num_hidden=3)
        self.value_crit = torch.nn.MSELoss()
        self.policy_crit = torch.nn.CrossEntropyLoss()
        self.opt = torch.optim.Adam(self.net.parameters(), lr=0.01, betas=(0.9, 0.999), weight_decay=0.4)
        with torch.no_grad():
            self.net.apply(init_weights)
        self.root = MCTS_NN_Node(board=game_board, net=self.net, prob=1)
        self.root.visits = 1
        
    def best(self):
        if self.board.curr_player_idx == 0:
            return max(self.root.children, key=lambda c: c[1].eval / c[1].visits)
        else:
            return min(self.root.children, key=lambda c: c[1].eval / c[1].visits)
        
    def static_train(self, epochs: int, X_train: np.ndarray, Y_train: np.ndarray, save_to: str, save_as: str = "net"):
        losses = []
        X_train = torch.from_numpy(X_train).float().to(self.net.device)
        Y_train = torch.from_numpy(Y_train).float().to(self.net.device)
        
        for epoch in range(epochs):
            self.opt.zero_grad()
            pred = self.net.forward(X_train)
            loss = self.crit.forward(pred, Y_train)
            losses.append(loss.item())
            loss.backward()
            self.opt.step()
        
        plt.plot(losses)
        plt.show()
        if save_to[-1] != '/':
            save_to += '/'
        torch.save(self.net, save_to + save_as + '.pt')
        
    def static_test(self, x: np.ndarray, y: np.ndarray, load: bool = False):
        if load:
            self.net = torch.load('value_head.pt')
        
        x = torch.from_numpy(x).float().to(self.net.device)
        y = torch.from_numpy(y).float().to(self.net.device)
        with torch.no_grad():
            pred: torch.Tensor = self.net.forward(x)
            loss = self.crit.forward(pred, y)
            pred = torch.clip(pred, min=0, max=1)
            print(f'Test Loss: {loss.item()}')
            cm = confusion_matrix.BinaryConfusionMatrix().to(self.net.device)
            acc = accuracy.Accuracy(task="binary").to(self.net.device)
            print(acc(pred, y))
            print(cm(pred, y))
        
    
    def train(self, self_learn_epochs: int, game_epochs: int, num_searches: int = 1000, tree_max_depth: int = -1, decay: float = 0.9, load: str = None, save: str = None):
        board_backup = deepcopy(self.board)
        path = '/'.join(self.board.__module__.split(".")[0:2]) + '/Models/'
        if load is not None:
            self.net = torch.load(path + load)
        losses = []
        
        for i in range(self_learn_epochs):
            self.board = deepcopy(board_backup)
            print(self.board)
            with torch.no_grad():
                while self.board.state == gameState.ONGOING:
                    self.calc_best_move(max_iter=num_searches, max_depth=tree_max_depth)
                    move, child = self.best()
                    
                    if move is not None:
                        self.move(move)
                    else:
                        print("ERROR")
                    print(self.board)
            print(self.board)
            print('winner is: ', self.board.winner)
            res = 0
            if self.board.state == gameState.DRAW:
                res = 0.5
            elif self.board.winner == self.board.players[0]:
                res = 1
            
            samples = self.board.encode()
            samples = samples.reshape((1, *samples.shape))

            value_labels = np.array([res])
            policy_labels = np.array([])
            value_labels = value_labels.reshape((1, *value_labels.shape))

            parent = child.parent
            while parent is not None:
                self.board.unmake_move()
                
                new_sample = self.board.encode()
                samples = np.concatenate([samples, new_sample.reshape((1, *new_sample.shape))])
                
                res = 2 * res - 1
                res = decay * res
                res = (res + 1) / 2
                new_label = np.array([res])
                value_labels = np.concatenate([value_labels, new_label.reshape((1, *new_label.shape))])
                
                parent = parent.parent
            
            samples = torch.tensor(samples).to(self.net.device)
            value_labels = torch.tensor(value_labels.astype(np.float32)).to(self.net.device)
            
            for epoch in range(game_epochs):
                self.opt.zero_grad()
                value_pred, policy_pred = self.net.forward(samples)
                value_loss = self.value_crit.forward(value_pred, value_labels)
                policy_loss = self.policy_crit.forward(policy_pred, policy_labels)
                loss = value_loss + policy_loss
                losses.append(loss.item())
                loss.backward()
                self.opt.step()
                
            self.root = MCTS_NN_Node(board=self.board, vh=self.net)
            self.root.eval = self.root.evaluate(self.board)
            self.root.visits = 1
        
        plt.plot(losses)
        plt.show()
        self.board = deepcopy(board_backup)
        torch.save(self.net, path + save)
        
    
    def calc_best_move(self, max_iter: int = 1000, max_depth = -1):
        max_d = 0
        
        for _ in range(max_iter):
            node = self.root
            board = deepcopy(self.board)
            depth = 0
            while len(node.untried_actions) == 0 and not node.is_leaf:
                (move, node) = node.select_child()
                board.make_move(move)
                depth += 1
            
            ev = node.eval
            if not node.is_leaf:
                if max_depth <= 1 or depth + 1 < max_depth:
                    (move, node) = node.expand(board)
                    board.make_move(move)
                    depth += 1
            
            node.backpropagate(ev)
            if depth > max_d:
                max_d = depth
                
                 
def init_weights(m):
    if type(m) == torch.nn.Linear:
        m.weight.fill_(0.001)
        m.bias.fill_(0.001)