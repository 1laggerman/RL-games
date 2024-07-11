import torch
from src.base import Game
import torch.nn.functional as F

torch.manual_seed(0)

class Model(torch.nn.Module):

    def __init__(self, board: Game, optimizer, critic) -> None:
        super().__init__()

        self.optim: torch.optim.Optimizer = optimizer
        self.criterion = critic

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def step(self, x: torch.Tensor) -> torch.Tensor:
        
        self.optim.zero_grad()
        res = self.forward(x)
        loss = self.criterion.forward(res, x)
        loss.backward()
        self.optim.step()

        return res, loss

