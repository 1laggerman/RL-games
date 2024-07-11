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

class MCTS_NN_Node(Node):
    policy: np.ndarray[float]
    self_prob: float
    
    def __init__(self, state: Game, net: torch.nn.Module, parent: "MCTS_NN_Node" = None, prob: float = 0) -> None:
        super(MCTS_NN_Node, self).__init__(state, parent=parent)
        self.net: BaseRenset = net
        self.tree_eval = 0
        self.final_eval = 0
        self.self_prob = prob
        self.visits = 1
        if state is not None:    
            with torch.no_grad():
                self.eval, self.policy = self.evaluate(state)
        
      
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
    
    
    # TODO: make sure node is not leaf by refrence
    def expand(self, board: Game) -> None:
        for move in self.untried_actions:
            prob = self.policy[board.map_move(move)]
            if prob > 0:
                b = deepcopy(board)
                b.make_move(move)
                if len(b.legal_moves) == 0:
                    print("start debugging")
                new_Node = MCTS_NN_Node(b, net=self.net, parent=self, prob=prob)
                if b.state == gameState.ENDED or b.state == gameState.DRAW:
                    new_Node.is_terminal = True
                new_child = (move, new_Node)
                self.children.append(new_child)
                
        self.untried_actions = []
        
        return random.choices(self.children, weights=self.policy[self.policy > 0], k=1)[0]
        
    def evaluate(self, board: Game) -> tuple[torch.Tensor, torch.Tensor]:
        value, policy = self.net.forward(torch.Tensor(board.encode()).unsqueeze(0).to(self.net.device))
        if self.player == board.players[1]:
            value = 1 - value
        
        # print(value.squeeze(0).detach().cpu().numpy()[0])
        policy = policy.squeeze(0).detach().cpu().numpy()
        legal = np.where(board.board == None, 1, 0).flatten()
        policy *= legal
        s = np.sum(policy)
        if s > 0:
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
    
class MCTS_NN_Tree(TreePlayer):
    root: MCTS_NN_Node
    
    def __init__(self, game_board: Game) -> None:
        super(MCTS_NN_Tree, self).__init__(game_board)
        self.net: BaseRenset = BaseRenset(game_board, num_resblocks=10, num_hidden=3)
        self.value_crit = torch.nn.MSELoss()
        self.policy_crit = torch.nn.CrossEntropyLoss()
        self.opt = torch.optim.Adam(self.net.parameters(), lr=0.001, betas=(0.9, 0.999), weight_decay=0.4)
        with torch.no_grad():
            self.net.apply(init_weights)
        
        
    def best(self):
        if self.game.curr_player_idx == 0:
            return max(self.root.children, key=lambda c: c[1].eval / c[1].visits)
        else:
            return min(self.root.children, key=lambda c: c[1].eval / c[1].visits)
        
    def create_node(self, state: Game, parent: Node = None) -> Node:
        return MCTS_NN_Node(state, parent=parent, net=self.net)
        
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
            
    def self_play(self, num_searches: int = 1000, tree_max_depth: int = -1, decay: float = 0.9) -> tuple[np.ndarray, np.ndarray]:
        board = deepcopy(self.game)
        node = deepcopy(self.root)
        with open('example.txt', 'a') as f:
            f.write(f'starting game\n{board}\n')
            print('starting game\n', board)
            with torch.no_grad():
                while board.state == gameState.ONGOING:
                    self.calc_best_move(max_iter=num_searches, max_depth=tree_max_depth, node=node, board=board)
                    probs = np.zeros((len(node.children)))
                    for i, child in enumerate(node.children):
                        probs[i] = child[1].visits
                    probs /= np.sum(probs)
                    if len(node.children) == 0:
                        print("ERROR - NO CHILDREN")
                        self.calc_best_move(max_iter=num_searches, max_depth=tree_max_depth, node=node, board=board)
                    move, node = random.choices(node.children, weights=probs, k=1)[0]
                    # move, child = self.best()
                    
                    if move is not None:
                        board.make_move(move)
                        # self.move(move)
                    else:
                        print("ERROR")
                    print(board)
                    f.write(f'{board}\n')
            print('winner is: ', board.winner)
        res = 0
        if board.state == gameState.DRAW:
            res = 0.5
        elif self.game.winner == board.players[0]:
            res = 1
        
        samples = board.encode()
        samples = samples.reshape((1, *samples.shape))

        value_labels = np.array([res])
        value_labels = value_labels.reshape((1, *value_labels.shape))
        
        policy_labels = np.array([node.policy])

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
    
    def train(self, self_learn_epochs: int, game_epochs: int, num_searches: int = 1000, tree_max_depth: int = -1, decay: float = 0.9, load: str = None, save: str = None):
        board_backup = deepcopy(self.game)
        path = '/'.join(self.game.__module__.split(".")[0:2]) + '/Models/'
        if load is not None:
            self.net: BaseRenset = torch.load(path + load)
        losses = []
        policy_losses = []
        value_losees = []
        
        for i in range(self_learn_epochs):
            samples, value_labels, policy_labels = self.self_play(num_searches, tree_max_depth, decay)
            
            samples = torch.tensor(samples).to(self.net.device)
            value_labels = torch.tensor(value_labels.astype(np.float32)).to(self.net.device)
            policy_labels = torch.tensor(policy_labels.astype(np.float32)).to(self.net.device)
            
            for epoch in range(game_epochs):
                self.opt.zero_grad()
                value_pred, policy_pred = self.net.forward(samples)
                value_loss = self.value_crit.forward(value_pred, value_labels)
                policy_loss = self.policy_crit.forward(policy_pred, policy_labels)
                policy_losses.append(policy_loss.item())
                value_losees.append(value_loss.item())
                loss = value_loss + policy_loss
                losses.append(loss.item())
                loss.backward()
                self.opt.step()
        
        # plt.plot(losses)
        # plt.title('total loss')
        
        # plt.plot(policy_losses)
        # plt.title('policy loss')
        
        plt.plot(value_losees)
        plt.title('value loss')
        
        plt.show()
        self.game = deepcopy(board_backup)
        torch.save(self.net, path + save)
        
    
    def calc_best_move(self, max_iter: int = 1000, max_depth = -1, node: MCTS_NN_Node = None, board: Game = None):
        if node is None:
            node = self.root
        if board is None:
            board = self.game
        max_d = 0
        
        if len(board.legal_moves) <= 2:
            print("start debuging")
        
        for _ in range(max_iter):
            running_node = node
            running_board = deepcopy(board)
            depth = 0
            while len(running_node.untried_actions) == 0 and not running_node.is_terminal:
                (move, running_node) = running_node.select_child()
                running_board.make_move(move)
                depth += 1
            
            if not running_node.is_terminal:
                if max_depth <= 1 or depth + 1 < max_depth:
                    (move, running_node) = running_node.expand(running_board)
                    running_board.make_move(move)
                    depth += 1
            
            running_node.backpropagate(running_node.eval)
            if depth > max_d:
                max_d = depth
                
                 
def init_weights(m):
    if type(m) == torch.nn.Linear:
        m.weight.fill_(0.001)
        m.bias.fill_(0.001)