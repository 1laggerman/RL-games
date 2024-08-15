from src.base import Move, Game, gameState, Network
from src.players.MCTS.Treeplayer import Node, TreePlayer, SArgs
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

class AZ_search_args(SArgs):
    def __init__(self, max_iters: int = 10, max_time: float = float('inf'), max_depth: int = -1) -> None:
        super().__init__(max_iters, max_time, max_depth)

class AZ_train_args:
    batch_size: int
    batch_epochs: int
    num_batches: int
    reward_decay: float

    def __init__(self, batch_size: int = 10, batch_epochs: int = 10, num_batches: int = 10, reward_decay: float = 0.9) -> None:
        self.batch_size = batch_size
        self.batch_epochs = batch_epochs
        self.num_batches = num_batches
        self.reward_decay = reward_decay

class AZ_NArgs:
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


class AZ_Data:
    X: np.ndarray
    policy_labels: np.ndarray
    value_labels: np.ndarray
    policy_preds: np.ndarray
    value_preds: np.ndarray
    
    def __init__(self, X: np.ndarray, policy_labels: np.ndarray, value_labels: np.ndarray, policy_preds: np.ndarray, value_preds: np.ndarray) -> None:
        self.X = X
        self.policy_labels = policy_labels
        self.value_labels = value_labels
        self.policy_preds = policy_preds
        self.value_preds = value_preds

    def concat(self, other: "AZ_Data") -> "AZ_Data":
        return AZ_Data(
            np.concatenate([self.X, other.X]),
            np.concatenate([self.policy_labels, other.policy_labels]),
            np.concatenate([self.value_labels, other.value_labels]),
            torch.cat([self.policy_preds, other.policy_preds]),
            torch.cat([self.value_preds, other.value_preds])
        )


# class AZ_Network(Network):

#     def evaluate(self):
#         value, policy = self.net.forward(torch.Tensor(game.encode()).unsqueeze(0).to(self.net.device))
        
#         policy = policy.squeeze(0).detach().cpu().numpy()
#         legal = np.where(game.board == None, 1, 0).flatten()
#         policy *= legal
#         s = np.sum(policy)
#         if s > 0:
#             policy /= np.sum(policy)

#         self.policy = policy
#         self.net_eval = value
#         return value.item()

#     def update(self):
#         pass

class Alpha_Zero_Node(Node):
    policy: np.ndarray[float]
    self_prob: float
    net_eval: torch.Tensor
    tree_eval: torch.Tensor

    def __init__(self, game: Game, net: torch.nn.Module, parent: "Alpha_Zero_Node" = None, prob: float = 0, maximizer: bool = True) -> None:
        self.net: BaseRenset = net
        self.self_prob = prob
        self.visits = 0
        self.maximizer = maximizer
        super(Alpha_Zero_Node, self).__init__(game, parent=parent)

    def select_child(self):
        
        # Exploration parameter
        C = math.sqrt(2) # TODO: check parameter

        # Calculate UCT score for each child and select the child with the highest score
        best_score = float('-inf')
        best_child = None
        
        # exploitation = math.sqrt(self.visits)
        for child in self.children:
            exploitation = child[1].eval / child[1].visits if self.maximizer else - child[1].eval / child[1].visits # TODO: check min-max
            exploration = math.sqrt(self.visits) / (1 + child[1].visits)
            uct_score = exploitation + C * exploration * child[1].self_prob

            if uct_score > best_score:
                best_score = uct_score
                best_child = child

        return best_child
    
    def evaluate(self, game: Game) -> float:
        if self.visits > 0:
            return self.net_eval.item()
        value, policy = self.net.forward(torch.Tensor(game.encode()).unsqueeze(0).to(self.net.device))
        
        legal = np.where(game.board == None, 1, 0).flatten()
        policy *= torch.tensor(legal)
        s = torch.sum(policy)
        if s > 0:
            policy /= s

        self.policy = policy.squeeze(0).detach().cpu().numpy()
        self.net_eval = value
        self.net_policy = policy
        return value.item()
    
    def update_rule(self, new_eval: float):
        self.eval += new_eval

    
class Alpha_Zero_player(TreePlayer):
    root: Alpha_Zero_Node
    net_args: AZ_NArgs
    def __init__(self, game: Game, name: str, search_args: AZ_search_args, net_args: AZ_NArgs) -> None:
        super(Alpha_Zero_player, self).__init__(game, name, search_args)
        self.net_args = net_args
        with torch.no_grad():
            self.net_args.net.apply(init_weights)
        
    def best(self) -> tuple[Move, Node]:
        if self.root.maximizer:
            return max(self.root.children, key=lambda c: c[1].eval / c[1].visits)
        else:
            return min(self.root.children, key=lambda c: c[1].eval / c[1].visits)
    
    def expand(self, game: Game, parent: Alpha_Zero_Node | None = None, move: Move | None = None) -> Node:
        prob = 0
        maximizer = True

        if parent is not None:
            maximizer = not parent.maximizer

            for move in parent.untried_actions:
                prob = parent.policy[game.map_move(move)]
                if prob > 0:
                    game.make_move(move)
                    new_Node = Alpha_Zero_Node(game, net=self.net_args.net, parent=parent, prob=prob, maximizer=maximizer)
                    # parent.backpropagate(new_Node.eval, stop_at=self.root)
                    new_child = (move, new_Node)
                    parent.children.append(new_child)
                    game.unmake_move()
            
            parent.untried_actions = []
            return parent
        else:
            return Alpha_Zero_Node(game, net=self.net_args.net, parent=parent, prob=prob, maximizer=maximizer)

    def load_model(self, name: str, path: str = '') -> None:
        if path == '':
            path = os.path.join(os.path.dirname(inspect.getfile(self.game.__class__)), "Models", name)

        optim_path = ''
        if path[-3:] != ".pt":
            path += ".pt"
            optim_path = path + "_optim.pt"
        else:
            optim_path = path[-3:] + "_optim.pt"
        
        self.net_args.net.load_state_dict(torch.load(path))
        self.net_args.optimizer.load_state_dict(torch.load(optim_path))

    def save_model(self, name: str, path: str = '', override: bool = False) -> None:
        if path == '':
            path = os.path.join(os.path.dirname(inspect.getfile(self.game.__class__)), "Models", name)

        optim_path = ''
        if path[-3:] != ".pt":
            path += ".pt"
            optim_path = path + "_optim.pt"
        else:
            optim_path = path[-3:] + "_optim.pt"

        base_file_name = path[-3:]
        if not override:
            i = 1
            while os.path.exists(path):
                path = f"{base_file_name}({i}).pt"
                optim_path = f"{base_file_name}_optim({i}).pt"
                i += 1
        torch.save(self.net_args.net.state_dict(), path)
        torch.save(self.net_args.optimizer.state_dict(), optim_path)
    
    def static_train(self, epochs: int, data: AZ_Data):
        losses = []
        value_losses = []
        policy_losses = []

        # TODO: try shuffled data
        # TODO: try train batches
        data.X = torch.from_numpy(data.X).float().to(self.net_args.device)
        data.policy_labels = torch.from_numpy(data.policy_labels).float().to(self.net_args.device)
        data.value_labels = torch.from_numpy(data.value_labels).float().to(self.net_args.device)

        for _ in range(epochs):
            self.net_args.optimizer.zero_grad()
            value, policy = self.net_args.net.forward(data.X)

            value_loss = self.net_args.value_crit.forward(value, data.value_labels)
            policy_loss = self.net_args.policy_crit.forward(policy, data.policy_labels)
            loss = value_loss + policy_loss

            losses.append(loss.item())
            value_losses.append(value_loss.item())
            policy_losses.append(policy_loss.item())

            loss.backward()
            self.net_args.optimizer.step()
        
        return losses, value_losses, policy_losses
        
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

    def self_play(self, decay: float = 0.9) -> AZ_Data:
        game = self.game
        first_time = True

        # with torch.no_grad():
        while game.state == gameState.ONGOING:
            move = self.get_move() # search the tree

            true_policy = np.zeros(self.root.policy.shape)
            for move, child in self.root.children:
                true_policy[game.map_move(move)] = child.visits
            true_policy /= np.sum(true_policy)

            if first_time:
                policy_labels = np.concatenate([true_policy.reshape((1, *true_policy.shape)),])
                policy_preds = self.root.net_policy
                first_time = False
            else:
                policy_labels = np.concatenate([policy_labels, true_policy.reshape((1, *true_policy.shape))])
                policy_preds = torch.cat((policy_preds, self.root.net_policy))
            
            move, node = random.choices(self.root.children, weights=self.root.policy[self.root.policy > 0], k=1)[0] # explore different moves for train

            if move is not None:
                game.make_move(move)
                self.update_state(move)
            else:
                print("ERROR")
            print(game)
        
        if game.state == gameState.DRAW:
            print('draw')
        else:
            print('winner is: ', game.winner.name)
        node: Alpha_Zero_Node | None

        res = game.players[0].reward
        
        samples = game.encode()
        samples = samples.reshape((1, *samples.shape))

        value_labels = np.array([res])
        value_labels = value_labels.reshape((1, *value_labels.shape))
        value_preds = node.net_eval

        node = node.parent
        while node is not None:
            game.unmake_move()
            
            new_sample = game.encode()
            
            res *= decay
            true_value = np.array([res])
            
            samples = np.concatenate([new_sample.reshape((1, *new_sample.shape)), samples])
            value_labels = np.concatenate([true_value.reshape((1, *true_value.shape)), value_labels])
            value_preds = torch.cat((value_preds, node.net_eval))

            node = node.parent

        policy_labels = np.concatenate([policy_labels, np.zeros((1, *self.root.policy.shape))])

        return AZ_Data(samples, policy_labels, value_labels, policy_preds, value_preds)
    
    def self_play_batch(self, batch_size: int = 10, decay: float = 0.9):
        self.reset_tree()
        data = self.self_play(decay)

        for _ in range(batch_size - 1):
            self.reset_tree()
            game_data = self.self_play(decay)

            data = data.concat(game_data) # TODO: test concatination

        return data
    
    def reset_tree(self):
        self.root = None
        self.start_node = None

    
    def train(self, train_args: AZ_train_args):
        losses, value_losses, policy_losses = [], [], []

        for i in range(train_args.num_batches):
            # get batch data
            data = self.self_play_batch(train_args.batch_size, train_args.reward_decay)
            # train
            l, vl, pl = self.static_train(train_args.batch_epochs, data)
            losses.extend(l)
            value_losses.extend(vl)
            policy_losses.extend(pl)
        
        plt.plot(losses, label='total loss')
        plt.plot(value_losses, label='value loss')
        plt.plot(policy_losses, label='policy loss')
        plt.show()
        


        


def init_weights(m):
    if type(m) == torch.nn.Linear:
        m.weight.fill_(0.001)
        m.bias.fill_(0.001)