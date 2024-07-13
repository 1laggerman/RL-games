from src.base import Move, Game, gameState
from src.players.MCTS.Treeplayer import Node, TreePlayer, searchArgs
from src.players.MCTS.Models.ML_architecture.resnet import BaseRenset
from pathlib import Path
import inspect

import math
import random
from copy import deepcopy
import numpy as np
import os

import torch
import matplotlib.pyplot as plt
from torchmetrics.classification import confusion_matrix, accuracy

torch.manual_seed(0)

class AZ_search_args(searchArgs):
    def __init__(self, max_iters: int = 10, max_time: float = float('inf'), max_depth: int = -1) -> None:
        super().__init__(max_iters, max_time, max_depth)

class AZ_net_args:
    net: torch.nn.Module
    device: torch.device
    optimizer: torch.optim.Optimizer
    value_crit: torch.nn
    policy_crit: torch.nn

    def __init__(self, net: torch.nn.Module, device: torch.device, optimizer: torch.optim.Optimizer, value_criterion: torch.nn, policy_criterion: torch.nn) -> None:
        self.net = net
        self.device = device
        self.optimizer = optimizer
        self.value_crit = value_criterion
        self.policy_crit = policy_criterion

class Alpha_Zero_Node(Node):
    policy: np.ndarray[float]
    self_prob: float
    net_eval: torch.Tensor
    tree_eval: torch.Tensor

    def __init__(self, game: Game, net: torch.nn.Module, parent: "Alpha_Zero_Node" = None, prob: float = 0, maximizer: bool = True) -> None:
        super(Alpha_Zero_Node, self).__init__(game, parent=parent)
        self.net: BaseRenset = net
        self.self_prob = prob
        self.visits = 1
        self.maximizer = maximizer

        if game is not None:    
            with torch.no_grad():
                self.net_eval, self.policy = self.evaluate(game)
                self.tree_eval = self.net_eval
                self.eval = self.net_eval.item()

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
    
    def expand(self, game: Game) -> None:
        for move in self.untried_actions:
            prob = self.policy[game.map_move(move)]
            if prob > 0:
                game.make_move(move)
                new_Node = Alpha_Zero_Node(game, net=self.net, parent=self, prob=prob)
                new_child = (move, new_Node)
                self.children.append(new_child)
                game.unmake_move()
                
        self.untried_actions = []
        
        return random.choices(self.children, weights=self.policy[self.policy > 0], k=1)[0]
    
    def evaluate(self, game: Game) -> tuple[torch.Tensor, torch.Tensor]:
        value, policy = self.net.forward(torch.Tensor(game.encode()).unsqueeze(0).to(self.net.device))
        
        policy = policy.squeeze(0).detach().cpu().numpy()
        legal = np.where(game.board == None, 1, 0).flatten()
        policy *= legal
        s = np.sum(policy)
        if s > 0:
            policy /= np.sum(policy)
        return value, policy
    
    def update_rule(self, new_eval: float):
        self.tree_eval += new_eval[0]

    
class Alpha_Zero_player(TreePlayer):
    root: Alpha_Zero_Node
    def __init__(self, game: Game, name: str, search_args: AZ_search_args, net_args: AZ_net_args) -> None:
        super(Alpha_Zero_player, self).__init__(game, name, search_args)
        self.net_args = net_args
        with torch.no_grad():
            self.net_args.net.apply(init_weights)
        
    def best(self) -> tuple[Move, Node]:
        if self.root.maximizer:
            return max(self.root.children, key=lambda c: c[1].eval / c[1].visits)
        else:
            return min(self.root.children, key=lambda c: c[1].eval / c[1].visits)
    
    def create_node(self, game: Game, parent: Alpha_Zero_Node = None, action: int = 0) -> Node:
        prob = parent.policy[action] if parent is not None else 0
        player_type = not parent.maximizer if parent is not None else True
        return Alpha_Zero_Node(game, net=self.net_args.net, parent=parent, prob=prob, maximizer=player_type)

    def load_model(self, name: str, path: str = '') -> None:
        if path == '':
            path = os.path.join(os.path.dirname(inspect.getfile(self.game.__class__)) / "Models", name)

        if path[-3:] != ".pt":
            path += ".pt"
        
        self.net_args.net.load_state_dict(torch.load(path))

    def save_model(self, name: str, path: str = '', override: bool = False) -> None:
        if path == '':
            path = os.path.join(os.path.dirname(inspect.getfile(self.game.__class__)) / "Models", name)

        if path[-3:] != ".pt":
            path += ".pt"

        base_file_name = path[-3:]
        if not override:
            i = 1
            while os.path.exists(path):
                path = f"{base_file_name}({i}).pt"
                i += 1
        torch.save(self.net_args.net.state_dict(), path)
    
    def static_train(self, epochs: int, X_train: np.ndarray, Y_train: np.ndarray, save_to: str, save_as: str = "net"):
        losses = []
        X_train = torch.from_numpy(X_train).float().to(self.net_args.device)
        Y_train = torch.from_numpy(Y_train).float().to(self.net_args.device)
        
        for _ in range(epochs):
            self.net_args.optimizer.zero_grad()
            pred = self.net_args.net.forward(X_train)
            loss = self.net_args.value_crit.forward(pred, Y_train)
            losses.append(loss.item())
            loss.backward()
            self.net_args.optimizer.step()
        
        plt.plot(losses)
        plt.show()
        if save_to[-1] != '/':
            save_to += '/'
        torch.save(self.net, save_to + save_as + '.pt')
        
    def static_test(self, x: np.ndarray, y: np.ndarray, load: bool = False):
        if load:
            self.net = torch.load('value_head.pt')
        
        x = torch.from_numpy(x).float().to(self.net_args.device)
        y = torch.from_numpy(y).float().to(self.net_args.device)
        with torch.no_grad():
            pred: torch.Tensor = self.net_args.net.forward(x)
            loss = self.net_args.value_crit.forward(pred, y)
            pred = torch.clip(pred, min=0, max=1)
            print(f'Test Loss: {loss.item()}')
            cm = confusion_matrix.BinaryConfusionMatrix().to(self.net_args.device)
            acc = accuracy.Accuracy(task="binary").to(self.net_args.device)
            print(acc(pred, y))
            print(cm(pred, y))

    def self_play(self, decay: float = 0.9):
        move_count = 0
        game = self.game
        with torch.no_grad():
            while game.state == gameState.ONGOING:
                move = self.get_move() # exploit method
                move, node = random.choices(self.root.children, weights=self.root.policy[self.root.policy > 0], k=1)[0] # explore method

                # if move_count == 0:
                #     move, node = self.root.children[2]
                #     move_count += 1
                # elif move_count == 1:
                #     for child in self.root.children:
                #         if child[0].dest_location == (2, 1):
                #             move, node = child
                #             break
                #     move_count += 1
                
                if move is not None:
                    game.make_move(move)
                    self.move(move)
                else:
                    print("ERROR")
                print(game)
        print('winner is: ', game.winner.name)

        res = game.players[0].reward
        
        samples = game.encode()
        samples = samples.reshape((1, *samples.shape))

        value_labels = np.array([res])
        value_labels = value_labels.reshape((1, *value_labels.shape))
        
        policy_labels = np.array([self.root.policy])

        node = node.parent
        while node is not None:
            game.unmake_move()
            
            new_sample = game.encode()
            
            res *= decay
            true_value = np.array([res])
            
            true_policy = np.zeros(node.policy.shape)
            for move, child in node.children:
                true_policy[game.map_move(move)] = child.visits
            true_policy /= np.sum(true_policy)
            
            samples = np.concatenate([samples, new_sample.reshape((1, *new_sample.shape))])
            value_labels = np.concatenate([value_labels, true_value.reshape((1, *true_value.shape))])
            policy_labels = np.concatenate([policy_labels, true_policy.reshape((1, *true_policy.shape))])
            
            node = node.parent
        
        return samples, value_labels, policy_labels
    
    # def train(self, self_learn_games: int, game_train_epochs: int, decay: float = 0.9):
        


def init_weights(m):
    if type(m) == torch.nn.Linear:
        m.weight.fill_(0.001)
        m.bias.fill_(0.001)