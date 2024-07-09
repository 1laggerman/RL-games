import torch
from src.base import Board
import torch.nn.functional as F

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
            # torch.nn.Softmax(dim=1)
            torch.nn.Tanh()
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
    
# -----------------------------------------------------------------------------------------------------------------------------

# class ResBlock(torch.nn.Module):
#     expansion = 1
    
#     def __init__(self, in_planes: int, out_planes: int, stride: int) -> None:
#         super(ResBlock, self).__init__()
#         self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#         self.conv1 = torch.nn.Conv2d(in_planes, out_planes, kernel_size=3, padding=1, stride=stride, bias=False, device=self.device)
#         self.bn1 = torch.nn.BatchNorm2d(out_planes, device=self.device)
#         self.conv2 = torch.nn.Conv2d(out_planes, out_planes, kernel_size=3, padding=1, stride=1, bias=False, device=self.device)
#         self.bn2 = torch.nn.BatchNorm2d(out_planes, device=self.device)
        
#         self.shortcut = torch.nn.Sequential()
#         if stride != 1 or in_planes != out_planes:
#             self.shortcut = torch.nn.Sequential(
#                 torch.nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False, device=self.device),
#                 torch.nn.BatchNorm2d(out_planes, device=self.device)
#             )
#         self.to(self.device)
        
        
#     def forward(self, x: torch.Tensor):
#         residual = x
#         x = F.relu(self.bn1(self.conv1(x)))
#         x = self.bn2(self.conv2(x))
#         x += self.shortcut(residual)
#         x = F.relu(x)
#         return x
    
# class ResnetConv(torch.nn.Module):
#     def __init__(self, block, num_resblocks: 'list[int]', board: Board) -> None:
#         super(ResnetConv, self).__init__()
#         self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
#         self.in_planes = 64

#         self.conv1 = torch.nn.Conv2d(board.encode().shape[0], 64, kernel_size=3, stride=1, padding=1, bias=False, device=self.device)
#         self.bn1 = torch.nn.BatchNorm2d(64, device=self.device)
#         self.layer1 = self._make_layer(block, 64, num_resblocks[0], stride=1)
#         self.layer2 = self._make_layer(block, 128, num_resblocks[1], stride=2)
#         self.layer3 = self._make_layer(block, 256, num_resblocks[2], stride=2)
#         self.layer4 = self._make_layer(block, 512, num_resblocks[3], stride=2)
        
#     def _make_layer(self, block, planes, num_blocks, stride):
#         strides = [stride] + [1]*(num_blocks-1)
#         layers = []
#         for stride in strides:
#             layers.append(block(self.in_planes, planes, stride))
#             self.in_planes = planes * block.expansion
#         return torch.nn.Sequential(*layers)
    
#     def forward(self, x):
#         x = F.relu(self.bn1(self.conv1(x)))
#         x = self.layer1(x)
#         x = self.layer2(x)
#         x = self.layer3(x)
#         x = self.layer4(x)
#         # x = F.avg_pool2d(x, 4)
#         return x

# class Renset(torch.nn.Module):
    
#     def __init__(self, block, num_resblocks: 'list[int]', num_classes) -> None:
#         super(Renset, self).__init__()
#         self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
#         self.conv = ResnetConv(block, num_resblocks)
        
#         self.linear = torch.nn.Linear(512 * 4 * 4 * block.expansion, num_classes, device=self.device)

#     def forward(self, x):
#         x = self.conv.forward(x)
#         x = x.view(x.size(0), -1)
#         x = self.linear(x)
#         return x