

class MCTSNode:
    visits: int = 0
    wins: int = 0
    parent: "MCTSNode"
    children: list[tuple[str, "MCTSNode"]]
    
    def __init__(self, parent: "MCTSNode") -> None:
        self.visits = 0
        self.wins = 0
        self.parent = parent
        self.children = list()
        