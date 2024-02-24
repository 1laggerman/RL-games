from connectFourNew import connect4_Move
# from abc import ABC, abstractmethod

# move = connect4_Move("Move Name", "Player Name", 3)
# m2 = connect4_Move("0", "")

# print(m1 == m2)

# class Move(ABC):
#     name: str = ""
#     player: str
    
#     def __init__(self, name: str, player: str) -> None:
#         super().__init__()
#         self.name = name
#         self.player = player
        
#     def __eq__(self, other: object) -> bool:
#         if isinstance(other, Move):
#             return self.name == other.name and self.player == other.player
#         return False

# class connect4_Move(Move):    
#     location: int
    
#     def __init__(self, name: str, player: str, loc: int) -> None:
#         super().__init__(name, player)
#         self.location = loc

# Creating an object of connect4_Move
move = connect4_Move("Move Name", "Player Name", 3)
