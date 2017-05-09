from sympy import *
from sympy.tensor import *
from sympy.tensor.array import Array

B, J = symbols('B J', cls=IndexedBase)
i0, i1, i2 = symbols('i0 i1 i2', cls=Idx)

def make_vararr(name, dim):
    v = [[Symbol(base + name + str(i)) for i in range(dim)] for base in ['v', 'd']]
    return Array(v)

def grad(U):
    d = U.shape[1]
    components = []
    for j in range(d):
        terms = [U[1 if (d-i == j+1) else 0, i] for i in range(d)]
        components.append(Mul(*terms))
    return Array(components)


U = make_vararr('u', 3)
V = make_vararr('v', 3)

gradu = grad(U)
gradv = grad(V)

Btgu = Array([sum(B[i0,i1,i2,i,j]*gradu[i] for i in range(3)) for j in range(3)])
Btgv = Array([sum(B[i0,i1,i2,i,j]*gradv[i] for i in range(3)) for j in range(3)])

result = sum(Btgu[i]*Btgv[i] for i in range(3))

print(result)
#print(factor(result))

