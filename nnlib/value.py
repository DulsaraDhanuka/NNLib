from ops import *

class Node:
    def __init__(self):
        self.grad: float = 1.0
        self.data: float = None
    def backward(self): raise NotImplementedError

class Value(Node):
    def __init__(self, data: float):
        super().__init__()
        self.data: float = data
    
    def backward(self): pass
