from ops import *
from value import Value
import random

'''

------------------------------
x1 = 1
w1 = 0

x2 = 1
w2 = 0

b = 0

out1 = x1w1 + x2w2 + b
out1 = out1.tanh()
out2 = x1w1 + x2w2 + b
out2 = out2.tanh()

L = (out1 - y1)^2 + (out2 - y2)^2

backprop L
update w1, w2, b

L = 3 + 2 * (4 * 2)
------------------------------------

Teaching an AND circuit
1, 1 -> 1
1, 0 -> 0
0, 1 -> 0
0, 0 -> 0

'''

'''
# ------------- Input neurons -------------
x1 = Value(1.0)
x2 = Value(1.0)

# ------------- Hidden neurons -------------
n1w1 = Value(0.0)
n1w2 = Value(0.0)
n1b = Value(0.0)
n1 = Add(Add(Mul(x1, n1w1), Mul(x1, n1w1)), n1b)
n1 = Tanh(n1)

n2w1 = Value(0.0)
n2w2 = Value(0.0)
n2b = Value(0.0)
n2 = Add(Add(Mul(x1, n2w1), Mul(x1, n2w1)), n2b)
n2 = Tanh(n2)

# ------------- Output neuron -------------
o1w1 = Value(0.0)
o1w1 = Value(0.0)
o1b = Value(0.0)
o1 = Add(Add(Mul(n1, o1w1), Mul(n1, o1w1)), o1b)
'''
def rand(): return random.randint(0, 100)/100

n1w1 = Value(rand())
n1w2 = Value(rand())
n1b = Value(rand())

n2w1 = Value(rand())
n2w2 = Value(rand())
n2b = Value(rand())

ow1 = Value(rand())
ow2 = Value(rand())
ob = Value(rand())

x1 = Value(1.0)
x2 = Value(1.0)
y = Value(1.0)

n1 = Add(Add(Mul(x1, n1w1), Mul(x2, n1w1)), n1b)
n1 = Tanh(n1)
n2 = Add(Add(Mul(x1, n2w1), Mul(x2, n2w1)), n2b)
n2 = Tanh(n2)
o = Add(Add(Mul(n1, ow1), Mul(n2, ow2)), ob)
print(o.data)

L = Pow(Sub(o, y), Value(2))
L.backward()

print(n1w1.grad)
print(n1w2.grad)
print(n1b.grad)

print(n2w1.grad)
print(n2w2.grad)
print(n2b.grad)

print(ow1.grad)
print(ow2.grad)
print(ob.grad)
