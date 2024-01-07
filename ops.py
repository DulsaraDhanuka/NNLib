from typing import Any
from value import Value, Node
import numpy as np

class Op(Node):
    def __init__(self) -> None:
        self._value = Value(0.0)
    
    def _evaluate(self) -> float: raise NotImplementedError
    def backward(self) -> None: raise NotImplementedError

    def __getattribute__(self, __name: str) -> Any:
        if __name == "data":
            self._value.data = self._evaluate()
            return self._value.data
        if __name == "grad": return self._value.grad
        return super().__getattribute__(__name)

    def __setattr__(self, __name: str, __value: Any) -> None:
        if __name == "grad": self._value.grad = __value; return
        super().__setattr__(__name, __value)

class BinaryOp(Op):
    def __init__(self, x: Node, y: Node) -> None:
        super().__init__()
        self._x: Node = x
        self._y: Node = y

class UnaryOp(Op):
    def __init__(self, x: Node) -> None:
        super().__init__()
        self._x: Node = x

class Neg(UnaryOp):
    def _evaluate(self):
        return -self._x.data
    
    def backward(self):
        self._x.grad = -self.grad
        self._x.backward()

class Tanh(UnaryOp):
    def _evaluate(self):
        return np.tanh(self._x.data)
    
    def backward(self):
        self._x.grad = (1 - np.tanh(self._x.data)**2) * self.grad
        self._x.backward()

class Add(BinaryOp):
    def _evaluate(self):
        return self._x.data + self._y.data

    def backward(self):
        self._x.grad = self.grad
        self._y.grad = self.grad
        self._x.backward()
        self._y.backward()

class Mul(BinaryOp):   
    def _evaluate(self):
        return self._x.data * self._y.data

    def backward(self):
        self._x.grad = self._y.data * self.grad
        self._y.grad = self._x.data * self.grad
        self._x.backward()
        self._y.backward()

class Pow(BinaryOp):
    def _evaluate(self):
        return self._x.data ** self._y.data

    def backward(self):
        self._x.grad = self._y.data * (self._x.data**(self._y.data - 1)) * self.grad
        self._y.grad = (self._x.data**self._y.data) * np.log(self._x.data) * self.grad
        self._x.backward()
        self._y.backward()

class Sub(BinaryOp):
    def _evaluate(self):
        return self._x.data - self._y.data

    def backward(self):
        self._x.grad = self.grad
        self._y.grad = -self.grad
        self._x.backward()
        self._y.backward()

class Div(BinaryOp):
    def _evaluate(self):
        return self._x.data / self._y.data

    def backward(self):
        self._x.grad = (1/self._y.data) * self.grad
        self._y.grad = (-self._x.data/(self._y.data**2)) * self.grad
        self._x.backward()
        self._y.backward()
