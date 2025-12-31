import numpy as np
from dataclasses import dataclass

# この辺の構造，将来的にデータベースに移行する

@dataclass
class Tree:
    id: int
    location: np.ndarray | None # (3, )
    dbh: float | None # 胸高直径

class Trees:
    trees: list[Tree]
    id_counter: int
    def __init__(self, trees: list[Tree] | None = None):
        if trees is None:
            trees = []
        self.trees = trees
        self.id_counter = 0
        for tree in trees:
            self.id_counter = max(self.id_counter, tree.id)
    
    def add_tree(self, location: np.ndarray | None = None, dbh: float | None = None) -> Tree:
        tree = Tree(id=self.id_counter + 1, location=location, dbh=dbh)
        self.id_counter += 1
        self.trees.append(tree)
        return tree
    
    def get_tree(self, id: int) -> Tree:
        return self.trees[id]