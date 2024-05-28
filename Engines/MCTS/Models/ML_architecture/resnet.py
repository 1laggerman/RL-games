import torch
from Games.Game import Board

torch.manual_seed(0)

class BaseRenset(torch.nn.Module):
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