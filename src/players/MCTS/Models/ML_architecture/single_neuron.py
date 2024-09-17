import torch
from src.base import Game
import torch.nn.functional as F

torch.manual_seed(0)

class neuron(torch.nn.Module):
    
    def __init__(self, board: Game) -> None:

        self.value = torch.nn.Linear(board.encode().shape[0], 1)
        shape = board.encode().shape
        
        self.backbone = torch.nn.Sequential(
            torch.nn.Linear()
        )

        self.policy_head = torch.nn.Sequential(
            torch.nn.Linear(32 * board.board.shape[0] * board.board.shape[1], len(board.all_moves), device=self.device),
            torch.nn.Softmax(dim=1)
        )
        
        self.value_head = torch.nn.Sequential(
        
            torch.nn.Linear(3 * board.board.shape[0] * board.board.shape[1], 1, device=self.device),
            torch.nn.Tanh()
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.tanh(self.fc(x))