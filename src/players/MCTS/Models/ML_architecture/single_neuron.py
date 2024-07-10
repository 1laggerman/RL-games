import torch
from src.base import Game
import torch.nn.functional as F

torch.manual_seed(0)

class neuron(torch.nn.Module):
    
    def __init__(self, board: Game) -> None:
        self.fc = torch.nn.Linear(board.encode().shape[0], 1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.tanh(self.fc(x))