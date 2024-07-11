from src.base import Move, Game, gameState
from src.players.MCTS.Treeplayer import Node, TreePlayer
from src.players.MCTS.Models.ML_architecture.resnet import BaseRenset

import math
import random
from copy import deepcopy
import numpy as np

import torch
import matplotlib.pyplot as plt
from torchmetrics.classification import confusion_matrix, accuracy

class Alpha_Zero_Node(Node):
    policy: np.ndarray[float]
    self_prob: float

    def __init__(self, game: Game, net: torch.nn.Module, parent: "Alpha_Zero_Node" = None, search_iters: int = 10, search_time: float = float('inf'), max_depth: int = -1, prob: float = 0, maximizer: bool = True) -> None:
        super(Alpha_Zero_Node, self).__init__(game, parent=parent)
        self.net: BaseRenset = net
        self.self_prob = prob
        self.visits = 1
        self.maximizer = maximizer

        if game is not None:    
            with torch.no_grad():
                self.eval, self.policy = self.evaluate(game)

    def select_child(self):
        
        # Exploration parameter
        C = math.sqrt(2)

        # Calculate UCT score for each child and select the child with the highest score
        best_score = float('-inf')
        best_child = None
        
        exploitation = math.sqrt(self.visits)
        for child in self.children:
            exploitation = 1 - (child[1].eval / (child[1].visits + 1))
            exploration = math.sqrt(self.visits) / (1 + child[1].visits)
            uct_score = exploitation + C * exploration * child[1].self_prob

            if uct_score > best_score:
                best_score = uct_score
                best_child = child

        return best_child  
    
    def expand(self, board: Game) -> None:
        for move in self.untried_actions:
            if move.dest_location == (0, 1):
                print("stop")
            prob = self.policy[board.map_move(move)]
            if prob > 0:
                board.make_move(move)
                new_Node = Alpha_Zero_Node(board, net=self.net, parent=self, prob=prob)
                new_child = (move, new_Node)
                self.children.append(new_child)
                board.unmake_move()
                
        self.untried_actions = []
        
        return random.choices(self.children, weights=self.policy[self.policy > 0], k=1)[0]
    
    def evaluate(self, board: Game) -> tuple[torch.Tensor, torch.Tensor]:
        value, policy = self.net.forward(torch.Tensor(board.encode()).unsqueeze(0).to(self.net.device))
        
        policy = policy.squeeze(0).detach().cpu().numpy()
        legal = np.where(board.board == None, 1, 0).flatten()
        policy *= legal
        s = np.sum(policy)
        if s > 0:
            policy /= np.sum(policy)
        return value, policy
    
    def update_rule(self, new_eval: float):
        self.eval += new_eval[0]

    
class Alpha_Zero_player(TreePlayer):
    root: Alpha_Zero_Node
    def __init__(self, game_board: Game, name: str, net: torch.nn.Module) -> None:
        super(Alpha_Zero_player, self).__init__(game_board, name)
        self.net = net
        self.value_crit = torch.nn.MSELoss()
        self.policy_crit = torch.nn.CrossEntropyLoss()
        self.opt = torch.optim.Adam(self.net.parameters(), lr=0.001, betas=(0.9, 0.999), weight_decay=0.4)
        with torch.no_grad():
            self.net.apply(init_weights)
        
    def best(self) -> tuple[Move, Node]:
        if self.root.maximizer:
            return max(self.root.children, key=lambda c: c[1].eval / c[1].visits)
        else:
            return min(self.root.children, key=lambda c: c[1].eval / c[1].visits)
    
    def create_node(self, game: Game, parent: Alpha_Zero_Node = None, action: int = 0) -> Node:
        prob = parent.policy[action] if parent is not None else 0
        player_type = not parent.maximizer if parent is not None else True
        return Alpha_Zero_Node(game, net=self.net, parent=parent, prob=prob, maximizer=player_type)
    
    def static_train(self, epochs: int, X_train: np.ndarray, Y_train: np.ndarray, save_to: str, save_as: str = "net"):
        losses = []
        X_train = torch.from_numpy(X_train).float().to(self.net.device)
        Y_train = torch.from_numpy(Y_train).float().to(self.net.device)
        
        for epoch in range(epochs):
            self.opt.zero_grad()
            pred = self.net.forward(X_train)
            loss = self.value_crit.forward(pred, Y_train)
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
            loss = self.value_crit.forward(pred, y)
            pred = torch.clip(pred, min=0, max=1)
            print(f'Test Loss: {loss.item()}')
            cm = confusion_matrix.BinaryConfusionMatrix().to(self.net.device)
            acc = accuracy.Accuracy(task="binary").to(self.net.device)
            print(acc(pred, y))
            print(cm(pred, y))

    def self_play(self, decay: float = 0.9):
        board = deepcopy(self.board)
        with torch.no_grad():
            while board.state == gameState.ONGOING:
                move = self.get_move() # exploit method
                move, node = random.choices(self.root.children, weights=self.root.policy, k=1)[0] # explore method
                
                if move is not None:
                    board.make_move(move)
                    self.move(move)
                else:
                    print("ERROR")
                print(board)
        print('winner is: ', board.winner)

        res = board.reward
        if board.winner != board.players[0]:
            res = -res
        
        samples = board.encode()
        samples = samples.reshape((1, *samples.shape))

        value_labels = np.array([res])
        value_labels = value_labels.reshape((1, *value_labels.shape))
        
        policy_labels = np.array([self.root.policy])

        node = node.parent
        while node is not None:
            board.unmake_move()
            
            new_sample = board.encode()
            
            res = 2 * res - 1
            res = decay * res
            res = (res + 1) / 2
            true_value = np.array([res])
            
            true_policy = np.zeros(node.policy.shape)
            for move, child in node.children:
                true_policy[board.map_move(move)] = child.visits
            true_policy /= np.sum(true_policy)
            
            samples = np.concatenate([samples, new_sample.reshape((1, *new_sample.shape))])
            value_labels = np.concatenate([value_labels, true_value.reshape((1, *true_value.shape))])
            policy_labels = np.concatenate([policy_labels, true_policy.reshape((1, *true_policy.shape))])
            
            node = node.parent
        
        return samples, value_labels, policy_labels


def init_weights(m):
    if type(m) == torch.nn.Linear:
        m.weight.fill_(0.001)
        m.bias.fill_(0.001)