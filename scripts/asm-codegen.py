from sympy import *
from sympy.tensor import *
from sympy.tensor.array import Array

B, J = symbols('B J', cls=IndexedBase)
i0, i1, i2 = symbols('i0 i1 i2', cls=Idx)

def make_vararr(name, dim):
    v = [[Symbol(base + name + str(i)) for i in range(dim)] for base in ['v', 'd']]
    return Array(v)

def make_vararr2(name, indices):
    dim = len(indices)
    return Array([
        [symbols('%s%s%s' % (deriv, name, i), cls=IndexedBase)[indices[i]] for i in range(dim)] for deriv in ('V', 'D')
    ])

def grad(U):
    d = U.shape[1]
    components = []
    for j in range(d):
        terms = [U[1 if (d-i == j+1) else 0, i] for i in range(d)]
        components.append(Mul(*terms))
    return Array(components)

dim = 3

I = (i0, i1, i2)[:dim]

U = make_vararr2('u', I)
V = make_vararr2('v', I)

if dim == 2:
    BI = Matrix([[B[i0,i1, i,j] for j in range(dim)] for i in range(dim)])
elif dim == 3:
    BI = Matrix([[B[i0,i1,i2, i,j] for j in range(dim)] for i in range(dim)])

gradu = grad(U)
gradv = grad(V)

Bgu = Matrix(BI.dot(gradu))
#Btgu = Matrix(BI.T.dot(gradu))
#Btgv = Matrix(BI.T.dot(gradv))

result = Bgu.T.dot(gradv)

print(result)
#print(factor(result))

