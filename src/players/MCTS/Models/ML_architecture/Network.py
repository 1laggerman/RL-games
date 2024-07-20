import torch
from src.base import Game
import torch.nn.functional as F

torch.manual_seed(0)

class Model(torch.nn.Module):

    def __init__(self, board: Game, backbone: torch.nn.Module, policy_preprocessor: torch.nn.Module, value_preprocessor: torch.nn.Module) -> None:
        super().__init__()
        self.value_head = torch.nn.Linear(value_preprocessor, 1)
        self.policy_head = torch.nn.Linear(board.encode().shape[0], len(board.legal_moves))

