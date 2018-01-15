from collections import OrderedDict, defaultdict
from functools import reduce
import operator
import numpy as np
import networkx
import copy
import numbers

class AsmVar:
    def __init__(self, name, src, shape, local, symmetric=False):
        self.name = name
        if isinstance(src, Expr):
            self.expr = src
            self.shape = src.shape
            self.src = None
        else:
            self.src = src
            self.expr = None
            assert shape is not None
            self.shape = shape
        self.local = local
        self.symmetric = (len(self.shape) == 2 and symmetric)
        self.as_expr = make_var_expr(self)

    def is_scalar(self):
        return self.shape is ()
    def is_vector(self):
        return len(self.shape) == 1
    def is_matrix(self):
        return len(self.shape) == 2

    def is_local(self):
        return self.local
    def is_field(self):
        return not self.local

class BasisFun:
    def __init__(self, name, vform, numcomp=None, component=None):
        self.name = name
        self.vform = vform
        self.numcomp = numcomp  # number of components; None means scalar
        self.component = component  # for vector-valued basis functions
        self.asmgen = None  # to be set to AsmGenerator for code generation

class VForm:
    """Abstract representation of a variational form."""
    def __init__(self, dim, vec=False, spacetime=False):
        self.dim = dim
        self.vec = vec
        self.spacetime = bool(spacetime)
        if self.spacetime:
            self.spacedims = range(self.dim - 1)
            self.timedim = self.dim - 1
        else:
            self.spacedims = range(self.dim)

        self.vars = OrderedDict()
        self.exprs = []         # expressions to be added to the result

        # predefined local variables with their generators (created on demand)
        self.predefined_vars = {
            'JacInv': lambda self: inv(self.Jac),
            'W':      lambda self: self.GaussWeight * abs(det(self.Jac)),
        }

    def basisfuns(self, parametric=False, components=(None,None)):
        def make_bfun_expr(bf):
            if bf.numcomp is not None:
                # return a vector which contains the components of the bfun
                return LiteralVectorExpr(
                    self.basisval(
                        BasisFun(bf.name, self, component=k),
                        physical=not parametric)
                    for k in self.spacedims)
            else:
                return self.basisval(bf, physical=not parametric)

        names = ('u', 'v')
        self.basis_funs = tuple(
                BasisFun(name, self, numcomp=nc)
                for (name,nc) in zip(names, components)
        )

        return tuple(make_bfun_expr(bf) for bf in self.basis_funs)

    def indices_to_D(self, indices):
        """Convert a list of derivative indices into a partial derivative tuple D."""
        D = self.dim * [0]
        for i in indices:
            D[i] += 1
        return tuple(D)

    def get_pderiv(self, bfun, indices=None, D=None):
        if D is None:
            D = self.indices_to_D(indices)
        name = '_d%s_%s' % (bfun.name, ''.join(str(k) for k in D))
        if not name in self.vars:
            self.let(name, PartialDerivExpr(bfun, D, physical=False))
        return self.vars[name].as_expr

    def get_pderivs(self, bfun, order):
        assert order == 1, 'only first derivatives implemented'
        return as_vector(self.get_pderiv(bfun, (i,)) for i in range(self.dim))

    def field_vars(self):
        return (var for var in self.vars.values() if var.is_field())

    def set_var(self, name, var):
        if name in self.vars:
            raise KeyError('variable %s already declared' % name)
        self.vars[name] = var
        return var

    def register_scalar_field(self, name, src=''):
        self.set_var(name, AsmVar(name, src=src, shape=(), local=False))

    def register_vector_field(self, name, size=None, src=''):
        if size is None: size = self.dim
        self.set_var(name, AsmVar(name, src=src, shape=(size,), local=False))

    def register_matrix_field(self, name, shape=None, symmetric=False, src=''):
        if shape is None: shape = (self.dim, self.dim)
        assert len(shape) == 2
        return self.set_var(name, AsmVar(name, src=src, shape=shape, local=False, symmetric=symmetric)).as_expr

    def declare_sourced_var(self, name, shape, src, symmetric=False):
        return self.set_var(name, AsmVar(name, src=src, shape=shape, local=True, symmetric=symmetric)).as_expr

    def add(self, expr):
        if self.vec:
            if expr.is_scalar():
                expr = self.substitute_vec_components(expr)
            if expr.is_matrix():
                expr = expr.ravel()
            if not expr.shape == (self.vec,):
                raise TypeError('vector assembler requires vector or matrix expression of proper length')
        else:
            if not expr.is_scalar():
                raise TypeError('require scalar expression')
        self.exprs.append(expr)

    def replace_vector_bfuns(self, expr, name, comp):
        bfun = expr.basisfun
        if bfun.name == name and bfun.component is not None:
            if bfun.component == comp:
                basic_bfun = [bf for bf in self.basis_funs if bf.name==name][0]
                return PartialDerivExpr(basic_bfun, expr.D, physical=expr.physical)
            else:
                return ConstExpr(0)

    def substitute_vec_components(self, expr):
        """Given a single scalar expression in terms of vector basis functions,
        return a matrix of scalar expressions where each basis function has
        been substituted by (u,0,..,0), ..., (0,...,0,u) successively (u being
        the underlying scalar basis function).
        """
        assert self.vec
        assert expr.is_scalar()
        assert len(self.basis_funs) == 2, 'Only implemented for bilinear forms'

        # for each output component, replace one component of the basis functions
        # by the corresponding scalar basis function and all others by 0
        result = []
        bfu, bfv = self.basis_funs
        assert self.vec == bfu.numcomp * bfv.numcomp, 'Incorrect output size'

        for i in range(bfv.numcomp):
            row = []
            for j in range(bfu.numcomp):
                exprij = copy.deepcopy(expr)    # transform_expr is destructive, so copy the original
                exprij = transform_expr(exprij,
                    lambda e: self.replace_vector_bfuns(e, bfv.name, i),
                    type=PartialDerivExpr)
                exprij = transform_expr(exprij,
                    lambda e: self.replace_vector_bfuns(e, bfu.name, j),
                    type=PartialDerivExpr)
                row.append(exprij)

            result.append(row)
        return as_matrix(result)

    def basisval(self, basisfun, physical=False):
        return PartialDerivExpr(basisfun, self.dim * (0,), physical=physical)

    def gradient(self, basisfun, dims=None, additional_derivs=None):
        if dims is None:
            dims = range(self.dim)
        if additional_derivs is None:
            additional_derivs = self.dim * [0]

        entries = []
        for k in dims:
            D = list(additional_derivs)
            D[k] += 1
            entries.append(PartialDerivExpr(basisfun, D))
        return LiteralVectorExpr(entries)

    def let(self, varname, expr, symmetric=False):
        var = AsmVar(varname, expr, shape=None, local=True, symmetric=symmetric)
        self.vars[varname] = var
        return var.as_expr

    # automatically produce caching getters for predefined on-demand local variables
    def __getattr__(self, name):
        if name in self.vars:
            return self.vars[name].as_expr
        elif name in self.predefined_vars:
            self.let(name, self.predefined_vars[name](self))
            return self.vars[name].as_expr
        else:
            msg = "'{0}' object has no attribute '{1}'"
            raise AttributeError(msg.format(type(self).__name__, name))

    # on-demand field variables (cannot currently be autogenerated)

    @property
    def GaussWeight(self):
        if not 'GaussWeight' in self.vars:
            self.declare_sourced_var('GaussWeight', shape=(), src='gauss_weights')
        return self.vars['GaussWeight'].as_expr

    @property
    def Jac(self):
        if not 'Jac' in self.vars:
            self.declare_sourced_var('Jac', shape=(self.dim,self.dim), src='geo_jac')
        return self.vars['Jac'].as_expr

    # expression analyis and transformations

    def dependency_graph(self):
        """Compute a directed graph of the dependencies between all used variables."""
        G = networkx.DiGraph()
        # make sure virtual basis function nodes are always in the graph
        G.add_nodes_from(('@u', '@v'))

        for e in self.all_exprs(type=VarExpr):
            var = e.var
            G.add_node(var)
            if var.expr:
                for dep in var.expr.depends():
                    G.add_edge(dep, var)
        return G

    def transitive_deps(self, dep_graph, vars):
        """Return all vars on which the given vars depend directly or indirectly, in linearized order."""
        return set().union(*(networkx.ancestors(dep_graph, var) for var in vars))

    def linearize_vars(self, vars):
        """Returns an admissible order for computing the given vars, i.e., which
        respects the dependency relation."""
        return [var for var in self.linear_deps if var in vars]

    def vars_without_dep_on(self, dep_graph, exclude):
        """Return a linearized list of all expr vars which do not depend on the given vars."""
        nodes = set(dep_graph.nodes())
        # remove 'exlude' vars and anything that depends on them
        for var in exclude:
            nodes.discard(var)
            nodes -= networkx.descendants(dep_graph, var)
        return self.linearize_vars(nodes)

    def dependency_analysis(self):
        dep_graph = self.dependency_graph()
        self.linear_deps = networkx.topological_sort(dep_graph)

        # determine precomputable vars (no dependency on basis functions)
        precomputable = self.vars_without_dep_on(dep_graph, ('@u', '@v'))
        self.precomp = [v for v in precomputable if v.expr]
        self.precomp_deps = [v for v in precomputable if v.src]

        for var in precomputable:
            # remove all dependencies since it's now precomputed
            # this ensures kernel_deps will not depend on dependencies of precomputed vars
            dep_graph.remove_edges_from(dep_graph.in_edges([var]))

        # compute linearized list of vars the kernel depends on
        kernel_deps = set().union(*(expr.depends() for expr in self.exprs))
        kernel_deps |= self.transitive_deps(dep_graph, kernel_deps)
        self.kernel_deps = self.linearize_vars(kernel_deps - {'@u', '@v'})

        # promote precomputed/manually sourced dependencies to field variables
        for var in self.kernel_deps:
            if var.src or var in self.precomp:
                var.local = False

        # separate precomp into locals (not used in kernel, only during precompute) and true dependencies
        self.precomp_locals = [v for v in self.precomp if v.local]
        self.precomp        = [v for v in self.precomp if not v.local]

    def all_exprs(self, type=None, once=True):
        """Deep, depth-first iteration of all expressions with dependencies.

        If `type` is given, only exprs which are instances of that type are yielded.
        If `once=True` (default), each expr is visited only once.
        """
        return iterexprs(self.exprs, deep=True, type=type, once=once)

    def transform(self, fun, type=None):
        """Apply `fun` to all exprs (or all exprs of the given `type`). If `fun` returns
        an expr, replace the old expr by this new one.
        """
        self.exprs = transform_exprs(self.exprs, fun, type=type, deep=True)

    def collect(self, type=None, filter=None):
        for e in self.all_exprs(type=type):
            if filter is None or filter(e):
                yield e

    def replace_physical_derivs(self, e):
        if not e.physical:
            return
        if sum(e.D) == 0:       # no derivatives?
            return e.make_parametric()

        if self.spacetime:
            # the following assumes a space-time cylinder -- can keep time derivatives parametric,
            # only need to transform space derivatives
            assert self.timedim == self.dim - 1 # to make sure the next line is correct
            D_x = e.D[:-1] + (0,)   # HACK: should be D[spacedims]; assume time is last
            if sum(D_x) == 0:
                return self.get_pderiv(e.basisfun, D=e.D)   # time derivatives are parametric
            elif sum(D_x) == 1:
                k = D_x.index(1)
                dts = e.D[-1] * (self.timedim,)
                spacegrad = as_vector(self.get_pderiv(e.basisfun, (i,) + dts)
                                      for i in self.spacedims)
                return inner(self.JacInv[self.spacedims, k], spacegrad)
        else:
            order = sum(e.D)
            if order == 1:
                k = e.D.index(1)    # get index of derivative direction
                return inner(self.JacInv[:, k], self.get_pderivs(e.basisfun, 1))

        assert False, 'higher order physical derivatives not implemented'

    def compute_recursive(self, func):
        values = {}
        for e in self.all_exprs():
            child_values = tuple(values[c] for c in e.children)
            values[e] = func(e, child_values)
        return values

    def extract_common_expressions(self):
        tmpidx = [0]
        def tmpname():
            tmpidx[0] += 1
            return '_tmp%i' % tmpidx[0]

        while True:
            # compute expression hashes and complexity
            hashes = self.compute_recursive(lambda e, child_hashes: e.hash(child_hashes))
            complexity = self.compute_recursive(lambda e, cc: e.base_complexity + sum(cc))

            # roots of expression trees for used variables and kernel expressions
            all_root_exprs = [e.var.expr for e in self.collect(type=VarExpr, filter=lambda e: e.var.expr)] + list(self.exprs)

            # Count occurrences of exprs according to their hashes.
            # Each var is visited once because of deep=False.
            # Subexpressions are counted according to their occurence due to once=False.
            expr_count = defaultdict(int)
            hash_to_exprs = {}
            for e in iterexprs(all_root_exprs, deep=False, once=False):
                h = hashes[e]
                expr_count[h] += 1
                hash_to_exprs[h] = e

            # find all nontrivial exprs which are used more than once
            cse = [e for (h,e) in hash_to_exprs.items()
                    if expr_count[h] > 1 and complexity[e] > 0]
            if not cse:
                break

            # find the most complex among all common subexprs
            biggest_cse = max(cse, key=complexity.get)
            h = hashes[biggest_cse]

            # extract it into a new variable
            var = self.let(tmpname(), biggest_cse)
            self.transform(lambda e: var if hashes[e] == h else None)
        return tmpidx[0] > 0    # did we do anything?

    def expand_mat_vec(self):
        """Convert all matrix and vector expressions into literal expressions, i.e.,
        into elementwise scalar expressions."""
        def expand(e):
            if e.is_var_expr():
                return
            if e.is_vector() and not isinstance(e, LiteralVectorExpr):
                return LiteralVectorExpr(e)
            if e.is_matrix() and not isinstance(e, LiteralMatrixExpr):
                return LiteralMatrixExpr(
                        [[e[i,j] for j in range(e.shape[1])]
                            for i in range(e.shape[0])])
        self.transform(expand)

    def finalize(self):
        """Performs standard transforms and dependency analysis."""
        # replace "dx" by quadrature weight function
        self.transform(lambda e: self.W, type=VolumeMeasureExpr)
        # replace physical derivs by proper expressions in terms of parametric derivs
        self.transform(self.replace_physical_derivs, type=PartialDerivExpr)
        # convert all expressions to scalar form
        self.expand_mat_vec()
        # fold constants, eliminate zeros
        self.transform(lambda e: e.fold_constants())
        # find common subexpressions and extract them into named variables
        self.extract_common_expressions()
        # perform dependency analysis for expressions and variables
        self.dependency_analysis()

    def find_max_deriv(self):
        return max(max(e.D) for e in self.all_exprs(type=PartialDerivExpr))


################################################################################
# Expressions for use in variational forms
################################################################################

class Expr:
    def __add__(self, other):  return OperExpr('+', self, other)
    def __radd__(self, other): return OperExpr('+', other, self)

    def __sub__(self, other):  return OperExpr('-', self, other)
    def __rsub__(self, other): return OperExpr('-', other, self)

    def __mul__(self, other):  return OperExpr('*', self, other)
    def __rmul__(self, other): return OperExpr('*', other, self)

    def __div__(self, other):  return OperExpr('/', self, other)
    def __rdiv__(self, other): return OperExpr('/', other, self)

    __truediv__ = __div__
    __rtruediv__ = __rdiv__

    def __pos__(self):  return self
    def __neg__(self):  return NegExpr(self)
    def __abs__(self):  return AbsExpr(self)

    def __bool__(self):  return True
    __nonzero__ = __bool__  # Python 2 compatibility

    def __len__(self):
        if self.is_scalar():
            raise TypeError('cannot get length of scalar')
        else:
            return self.shape[0]

    # convenience accessors for child nodes
    @property
    def x(self): return self.children[0]
    @property
    def y(self): return self.children[1]
    @property
    def z(self): return self.children[2]

    def is_scalar(self):    return self.shape is ()
    def is_vector(self):    return len(self.shape) == 1
    def is_matrix(self):    return len(self.shape) == 2
    def depends(self):
        return set().union(*(x.depends() for x in self.children))

    def is_var_expr(self):
        return False

    def dx(self, k, times=1):
        return Dx(self, k, times)
    def dt(self, times=1):
        return Dt(self, times)
    def dot(self, x):
        return dot(self, x)
    __matmul__ = dot        # allow @ operator in Python 3.5+
    @property
    def T(self):
        if not self.is_matrix():
            raise TypeError('can only transpose matrices')
        return TransposedMatrixExpr(self)
    def ravel(self):
        if not self.is_matrix():
            raise TypeError('can only ravel matrices')
        return LiteralVectorExpr(self[i,j] for i in range(self.shape[0])
                                           for j in range(self.shape[1]))

    def __getitem__(self, I):
        """Default __getitem__() implementations. Child classes need only implement at()."""
        if self.is_scalar():
            raise TypeError('cannot index scalar expression')
        elif self.is_vector():
            return self._default_vector_getitem(I)
        else:  # matrix
            return self._default_matrix_getitem(I)

    def _default_vector_getitem(self, i):
        i = _to_indices(i, self.shape[0])
        if np.isscalar(i):
            return self.at(i)
        else:
            return LiteralVectorExpr(self.at(ii) for ii in i)

    def _default_matrix_getitem(self, ij):
        i = _to_indices(ij[0], self.shape[0])
        j = _to_indices(ij[1], self.shape[1])
        sca_i, sca_j = np.isscalar(i), np.isscalar(j)
        if sca_i and sca_j:
            return self.at(i, j)
        elif sca_i and not sca_j:
            return LiteralVectorExpr(self.at(i, jj) for jj in j)
        elif not sca_i and sca_j:
            return LiteralVectorExpr(self.at(ii, j) for ii in i)
        else:
            return LiteralMatrixExpr([[self.at(ii, jj) for jj in j]
                                                       for ii in i])

    def hash_key(self):
        return ()

    def hash(self, child_hashes):
        return hash((type(self), self.shape) + self.hash_key() + child_hashes)

    def is_constant(self, val):
        return False

    def is_zero(self):
        return self.is_constant(0)

    def fold_constants(self):
        return self

    def __deepcopy__(self, memo):
        new = copy.copy(self)
        new.children = copy.deepcopy(self.children, memo)
        return new

    base_complexity = 1

def make_var_expr(var):
    """Create an expression of the proper shape which refers to the variable `var`."""
    shape = var.shape
    if shape is ():
        return ScalarVarExpr(var)
    elif len(shape) == 1:
        return VectorVarExpr(var)
    elif len(shape) == 2:
        return MatrixVarExpr(var)
    else:
        assert False, 'invalid shape'

class ConstExpr(Expr):
    def __init__(self, value):
        self.shape = ()
        self.value = float(value)
        self.children = ()
    def __str__(self):
        return str(self.value)
    def is_constant(self, val):
        return abs(self.value - val) < 1e-15
    def gencode(self):
        return repr(self.value)
    base_complexity = 0

class VarExpr(Expr):
    """Abstract base class for exprs which refer to named variables."""
    def is_var_expr(self):
        return True
    def __str__(self):
        return self.var.name
    def depends(self):
        return set((self.var,))
    def hash_key(self):
        return (self.var.name,)
    base_complexity = 0

class ScalarVarExpr(VarExpr):
    def __init__(self, var):
        self.var = var
        self.shape = ()
        self.children = ()
    def gencode(self):
        return self.var.name

class VectorVarExpr(VarExpr):
    def __init__(self, var):
        self.var = var
        self.shape = var.shape
        assert len(self.shape) == 1
        self.children = ()
    def at(self, i):
        return VectorEntryExpr(self, i)

class MatrixVarExpr(VarExpr):
    """Matrix expression which is represented by a matrix reference and shape."""
    def __init__(self, var):
        self.var = var
        self.shape = tuple(var.shape)
        assert len(self.shape) == 2
        self.symmetric = var.symmetric
        self.children = ()
    def at(self, i, j):
        return MatrixEntryExpr(self, i, j)

class LiteralVectorExpr(Expr):
    """Vector expression which is represented by a list of individual expressions."""
    def __init__(self, entries):
        entries = tuple(as_expr(e) for e in entries)
        self.shape = (len(entries),)
        self.children = entries
        if not all(e.is_scalar() for e in self.children):
            raise ValueError('all vector entries should be scalars')
    def __str__(self):
        return '(' + ', '.join(str(c) for c in self.children) + ')'
    def at(self, i):
        return self.children[i]
    base_complexity = 0

class LiteralMatrixExpr(Expr):
    """Matrix expression which is represented by a 2D array of individual expressions."""
    def __init__(self, entries):
        entries = np.array(entries, dtype=object)
        self.shape = entries.shape
        self.children = tuple(as_expr(e) for e in entries.flat)
        if not all(e.is_scalar() for e in self.children):
            raise ValueError('all matrix entries should be scalars')
    def at(self, i, j):
        return self.children[i * self.shape[1] + j]
    base_complexity = 0

class VectorEntryExpr(Expr):
    def __init__(self, x, i):
        assert isinstance(x, VectorVarExpr)   # can only index named vectors
        self.shape = ()
        assert x.is_vector(), 'indexed expression is not a vector'
        self.i = int(i)
        self.children = (x,)
    def __str__(self):
        return '%s[%i]' % (self.x.var.name, self.i)
    def hash_key(self):
        return (self.i,)
    def gencode(self):
        return '{x}[{i}]'.format(x=self.x.var.name, i=self.i)
    base_complexity = 0

class MatrixEntryExpr(Expr):
    def __init__(self, mat, i, j):
        assert isinstance(mat, MatrixVarExpr)
        self.shape = ()
        self.i = i
        self.j = j
        self.children = (mat,)
    def __str__(self):
        return '%s[%i,%i]' % (self.x.var.name, self.i, self.j)
    def to_seq(self, i, j):
        if self.x.symmetric and i > j:
            i, j = j, i
        return i * self.x.shape[0] + j
    def hash_key(self):
        return (self.i, self.j)
    def gencode(self):
        return '{name}[{k}]'.format(name=self.x.var.name,
                k=self.to_seq(self.i, self.j))
    base_complexity = 0

class TransposedMatrixExpr(Expr):
    def __init__(self, mat):
        if not mat.is_matrix(): raise TypeError('can only transpose matrices')
        self.shape = (mat.shape[1], mat.shape[0])
        self.children = (mat,)
    def __str__(self):
        return 'transpose(%s)' % self.x
    def at(self, i, j):
        return self.x[j, i]

class BroadcastExpr(Expr):
    """Simple broadcasting from scalars to arbitrary shapes."""
    def __init__(self, expr, shape):
        self.shape = shape
        self.children = (expr,)
    def __str__(self):
        return str(self.x)
    def at(self, *I):
        return self.x

class NegExpr(Expr):
    def __init__(self, expr):
        if not expr.is_scalar(): raise TypeError('can only negate scalars')
        self.shape = ()
        self.children = (expr,)
    def __str__(self):
        return '-%s' % str(self.x)
    def gencode(self):
        return '-' + self.x.gencode()
    base_complexity = 0 # don't bother extracting subexpressions which are simple negation

class AbsExpr(Expr):
    def __init__(self, expr):
        if not expr.is_scalar(): raise TypeError('can only take abs of scalars')
        self.shape = ()
        self.children = (expr,)
    def __str__(self):
        return 'abs(%s)' % str(self.x)
    def gencode(self):
        return 'fabs(%s)' % self.x.gencode()

def OperExpr(oper, x, y):
    # coerce arguments to Expr, in case they are number literals
    x = as_expr(x)
    y = as_expr(y)

    if x.is_scalar() and y.is_scalar():
        return ScalarOperExpr(oper, x, y)
    elif len(x.shape) == len(y.shape):      # vec.vec or mat.mat
        return TensorOperExpr(oper, x, y)
    elif x.is_scalar() and not y.is_scalar():
        return OperExpr(oper, BroadcastExpr(x, y.shape), y)
    elif not x.is_scalar() and y.is_scalar():
        return OperExpr(oper, x, BroadcastExpr(y, x.shape))
    else:
        raise TypeError('operation not implemented for shapes: {} {} {}'.format(oper, x.shape, y.shape))

class ScalarOperExpr(Expr):
    def __init__(self, oper, x, y):
        assert x.is_scalar() and y.is_scalar(), 'expected scalars'
        assert x.shape == y.shape
        self.shape = x.shape
        self.oper = oper
        self.children = (x,y)

    def __str__(self):
        return '%s(%s)' % (self.oper, ', '.join(str(c) for c in self.children))

    def hash_key(self):
        return (self.oper,)

    def fold_constants(self):
        # if only constants, compute the resulting value
        if all(isinstance(c, ConstExpr) for c in self.children):
            func = _oper_to_func[self.oper]
            return ConstExpr(reduce(func, (c.value for c in self.children)))

        # else, check for zeros and negations
        if self.oper == '+':
            if self.x.is_zero():            # 0 + y  -->  y
                return self.y
            if self.y.is_zero():            # x + 0  -->  x
                return self.x
            if isinstance(self.y, NegExpr): # x + (-y)  -->  x - y
                return OperExpr('-', self.x, self.y.x)
        elif self.oper == '-':
            if self.x.is_zero():            # 0 - y  -->  -y
                return -self.y
            if self.y.is_zero():            # x - 0  -->  x
                return self.x
            if isinstance(self.y, NegExpr): # x - (-y)  -->  x + y
                return OperExpr('+', self.x, self.y.x)
        elif self.oper == '*':
            if any(c.is_zero() for c in self.children):
                return ConstExpr(0)
            if self.x.is_constant(1):       # 1 * y  -->  y
                return self.y
            if self.x.is_constant(-1):      # -1 * y -->  -y
                return -self.y
            if self.y.is_constant(1):       # x * 1  -->  x
                return self.x
            if self.y.is_constant(-1):      # x * -1  -->  -x
                return -self.x
        elif self.oper == '/':
            if self.x.is_zero():            # 0 / y  -->  0
                return ConstExpr(0)
            if self.y.is_constant(1):       # x / 1  -->  x
                return self.x
            if self.y.is_constant(-1):      # x / -1  -->  -x
                return -self.x
            if self.y.is_zero():            # x / 0  -->  ERROR
                raise ZeroDivisionError('division by zero in expr %s' % self)
        return self

    def gencode(self):
        sep = ' ' + self.oper + ' '
        return '(' + sep.join(x.gencode() for x in self.children) + ')'

_oper_to_func = {
        '+': operator.add,
        '-': operator.sub,
        '*': operator.mul,
        '/': operator.truediv,
}

class TensorOperExpr(Expr):
    def __init__(self, oper, x, y):
        assert x.shape == y.shape, 'incompatible shapes'
        self.shape = x.shape
        self.oper = oper
        self.children = (x,y)

    def __str__(self):
        return '%s(%s)' % (self.oper, ', '.join(str(c) for c in self.children))

    def at(self, *I):
        func = _oper_to_func[self.oper]
        return reduce(func, (z[I] for z in self.children))

    def hash_key(self):
        return (self.oper,)

class VectorCrossExpr(Expr):
    def __init__(self, x, y):
        assert x.shape == y.shape == (3,), 'cross() requires 3D vectors'
        self.shape = x.shape
        self.children = (x,y)

    def at(self, i):
        if   i == 0:  return self.x[1]*self.y[2] - self.x[2]*self.y[1]
        elif i == 1:  return self.x[2]*self.y[0] - self.x[0]*self.y[2]
        elif i == 2:  return self.x[0]*self.y[1] - self.x[1]*self.y[0]
        else:         raise IndexError('invalid index %s, should be 0, 1, or 2' % i)

class OuterProdExpr(Expr):
    def __init__(self, x, y):
        if not (x.is_vector() and y.is_vector()):
            raise TypeError('outer() requires two vectors')
        self.shape = (len(x), len(y))
        self.children = (x,y)
    def at(self, i, j):
        return self.x[i] * self.y[j]

class PartialDerivExpr(Expr):
    """A scalar expression which refers to the value of a basis function or one
    of its partial derivatives."""
    def __init__(self, basisfun, D, physical=False):
        self.shape = ()
        self.basisfun = basisfun
        self.D = tuple(D)
        self.children = ()
        self.physical = bool(physical)

    def __str__(self):
        s = self.basisfun.name
        if self.basisfun.component is not None:
            s += '[%d]' % self.basisfun.component
        if sum(self.D) != 0:
            s += '_' + ''.join(str(k) for k in self.D)
            s += '(phys)' if self.physical else '(para)'
        return s

    def hash_key(self):
        return (self.basisfun, self.D, self.physical)

    def gencode(self):
        assert not self.physical, 'cannot generate code for physical derivative'
        return self.basisfun.asmgen.gen_pderiv(self.basisfun, self.D)
    def depends(self):
        return set(('@' + self.basisfun.name,))

    def make_parametric(self):
        if not self.physical: raise ValueError('derivative is already parametric')
        return PartialDerivExpr(self.basisfun, self.D, physical=False)

    def dx(self, k, times=1):
        Dnew = list(self.D)
        Dnew[k] += times
        return PartialDerivExpr(self.basisfun, Dnew, physical=self.physical)


def _indices_from_slice(sl, n):
    start = sl.start
    if start is None: start = 0
    if start < 0: start += n
    stop = sl.stop
    if stop is None: stop = n
    if stop < 0: stop += n
    step = sl.step
    if step is None: step = 1
    return tuple(range(start, stop, step))

def _to_indices(x, n):
    if isinstance(x, slice):
        return _indices_from_slice(x, n)
    elif np.isscalar(x):
        if x < 0: x += n
        if 0 <= x < n:
            return x
        else:
            raise IndexError
    else:
        return tuple(x)


class MatVecExpr(Expr):
    def __init__(self, A, x):
        assert A.is_matrix() and x.is_vector()
        if not A.shape[1] == x.shape[0]:
            raise ValueError('incompatible shapes: %s, %s' % (A.shape, x.shape))
        self.shape = (A.shape[0],)
        self.children = (A, x)
    def __str__(self):
        return 'dot(%s, %s)' % (self.x, self.y)
    def at(self, i):
        return reduce(operator.add,
            (self.x[i, j] * self.y[j] for j in range(self.y.shape[0])))

class MatMatExpr(Expr):
    def __init__(self, A, B):
        assert A.is_matrix() and B.is_matrix()
        if not A.shape[1] == B.shape[0]:
            raise ValueError('incompatible shapes: %s, %s' % (A.shape, B.shape))
        self.shape = (A.shape[0], B.shape[1])
        self.children = (A, B)
    def __str__(self):
        return 'dot(%s, %s)' % (self.x, self.y)
    def at(self, i, j):
        return reduce(operator.add,
            (self.x[i, k] * self.y[k, j] for k in range(self.x.shape[1])))

class VolumeMeasureExpr(Expr):
    def __init__(self):
        self.shape = ()
        self.children = ()
    def __str__(self):
        return 'dx'

# expression utility functions #################################################

def iterexprs(exprs, deep=False, type=None, once=True):
    """Iterate through all subexpressions of the list of expressions `exprs` in depth-first order.

    If `deep=True`, follow variable references.
    If `type` is given, only exprs which are instances of that type are yielded.
    If `once=True` (default), each expr is visited only once.
    """
    seen = set()    # remember which nodes we've visited already
    def recurse(e):
        if once:
            if e in seen:
                return
            else:
                seen.add(e)

        for c in e.children:
            yield from recurse(c)
        if (deep and e.is_var_expr()
                 and e.var.expr is not None):
            yield from recurse(e.var.expr)
        if type is None or isinstance(e, type):
            yield e
    for e in exprs:
        yield from recurse(e)

def mapexprs(exprs, fun, deep=False):
    """Replace each expr `e` in a list of expr trees by `fun(e)`, depth first.

    If `deep=True`, follow variable references.
    """
    seen = set()    # remember the nodes whose children we've transformed already

    def recurse_children(e):
        if e not in seen:
            e.children = recurse(e.children)
            seen.add(e)

    def recurse(es):
        result = []
        for e in es:
            if (deep and e.is_var_expr()
                     and e.var.expr is not None):
                var = e.var
                recurse_children(var.expr)
                var.expr = fun(var.expr)
                result.append(e)
            else:
                recurse_children(e)
                result.append(fun(e))
        return tuple(result)
    return recurse(exprs)

def make_applyfun(fun, type):
    def applyfun(e):
        e2 = None
        if type is None or isinstance(e, type):
            e2 = fun(e)
        return e2 if e2 is not None else e
    return applyfun

def transform_exprs(exprs, fun, type=None, deep=False):
    fun = make_applyfun(fun, type)
    return mapexprs(exprs, fun, deep=deep)

def transform_expr(expr, fun, type=None, deep=False):
    return transform_exprs((expr,), fun, type=type, deep=deep)[0]

def tree_print(expr, data=None, indent=''):
    stop = False
    if hasattr(expr, 'oper'):
        s = '(%s)' % expr.oper
    elif isinstance(expr, VectorEntryExpr) or isinstance(expr, MatrixEntryExpr):
        s = str(expr)
        stop = True
    elif expr.is_var_expr() or isinstance(expr, VolumeMeasureExpr) or isinstance(expr, PartialDerivExpr) or isinstance(expr, ConstExpr):
        s = str(expr)
    else:
        s = type(expr).__name__

    if data is None:
        print(indent + s)
    else:
        print(indent + s + ' (%s)' % data[expr])
    if not stop:
        for c in expr.children:
            tree_print(c, data, indent + '  ')

# expression manipulation functions ############################################
# notation is as close to UFL as possible

dx = VolumeMeasureExpr()

def Dx(expr, k, times=1):
    if expr.is_var_expr():
        expr = expr.var.expr    # access underlying expression - mild hack
    if expr.is_vector():
        return LiteralVectorExpr(Dx(z, k, times) for z in expr)
    elif expr.is_matrix():
        raise NotImplementedError('derivative of matrix not implemented')
    else:   # scalar
        if not isinstance(expr, PartialDerivExpr):
            raise TypeError('can only compute derivatives of basis functions')
        return expr.dx(k, times)

def Dt(expr, times=1):
    if expr.is_var_expr():
        expr = expr.var.expr    # access underlying expression - mild hack
    if expr.is_vector():
        return LiteralVectorExpr(Dt(z, times) for z in expr)
    elif expr.is_matrix():
        raise NotImplementedError('time derivative of matrix not implemented')
    else:   # scalar
        if not isinstance(expr, PartialDerivExpr):
            raise TypeError('can only compute derivatives of basis functions')
        if not expr.basisfun.vform.spacetime:
            raise Exception('can only compute time derivatives in spacetime assemblers')
        return expr.dx(expr.basisfun.vform.timedim, times)

def grad(expr, dims=None):
    if expr.is_var_expr():
        expr = expr.var.expr    # access underlying expression - mild hack
    if expr.is_vector():
        return as_matrix([grad(z, dims=dims) for z in expr])  # compute Jacobian of vector expression
    if not isinstance(expr, PartialDerivExpr):
        raise TypeError('can only compute gradient of basis function')
    if dims is None:
        dims = expr.basisfun.vform.spacedims
    return LiteralVectorExpr(Dx(expr, k) for k in dims)

def div(expr):
    if not expr.is_vector():
        raise TypeError('can only compute divergence of vector expression')
    return tr(grad(expr))

def as_expr(x):
    if isinstance(x, Expr):
        return x
    elif isinstance(x, numbers.Real):
        return ConstExpr(x)
    else:
        raise TypeError('cannot coerce %s to expression' % x)

def as_vector(x): return LiteralVectorExpr(x)
def as_matrix(x): return LiteralMatrixExpr(x)

def inner(x, y):
    if not x.is_vector() or x.is_matrix():
        raise TypeError('inner() requires vector or matrix expressions')
    if not x.shape == y.shape:
        raise ValueError('incompatible shapes in inner product')
    if x.is_vector():
        return reduce(operator.add, (x[i] * y[i] for i in range(x.shape[0])))
    else:   # matrix
        return reduce(operator.add,
                (x[i,j] * y[i,j] for i in range(x.shape[0])
                                 for j in range(x.shape[1])))

def dot(a, b):
    if a.is_vector() and b.is_vector():
        return inner(a, b)
    elif a.is_matrix() and b.is_vector():
        return MatVecExpr(a, b)
    elif a.is_matrix() and b.is_matrix():
        return MatMatExpr(a, b)
    else:
        raise TypeError('invalid types in dot')

def tr(A):
    """Trace of a matrix."""
    if not A.is_matrix() or A.shape[0] != A.shape[1]:
        raise ValueError('can only compute trace of square matrices')
    return reduce(operator.add, (A[i,i] for i in range(A.shape[0])))

def minor(A, i, j):
    m, n = A.shape
    B = [[A[ii,jj] for jj in range(n) if jj != j]
            for ii in range(m) if ii != i]
    return det(as_matrix(B))

def det(A):
    """Determinant of a matrix."""
    if not A.is_matrix() or A.shape[0] != A.shape[1]:
        raise ValueError('can only compute determinant of square matrices')
    n = A.shape[0]
    if n == 0:
        return ConstExpr(1)
    elif n == 1:
        return A[0,0]
    else:
        return reduce(operator.add,
                ((-1)**j * (A[0,j] * minor(A, 0, j))
                    for j in range(n)))

def inv(A):
    """Inverse of a matrix."""
    if not A.is_matrix() or A.shape[0] != A.shape[1]:
        raise ValueError('can only compute inverse of square matrices')
    n = A.shape[0]
    invdet = ConstExpr(1) / det(A)
    cofacs = as_matrix(
            [[ (-1)**(i+j) * minor(A, i, j)
                for i in range(n)]
                for j in range(n)])
    return invdet * cofacs

def cross(x, y):
    return VectorCrossExpr(x, y)

def outer(x, y):
    return OuterProdExpr(x, y)

################################################################################
# concrete variational forms
################################################################################

def mass_vf(dim):
    V = VForm(dim)
    u, v = V.basisfuns()
    V.add(u * v * dx)
    return V

def stiffness_vf(dim):
    V = VForm(dim)
    u, v = V.basisfuns(parametric=True)
    B = V.let('B', V.W * dot(V.JacInv, V.JacInv.T), symmetric=True)
    V.add(B.dot(grad(u)).dot(grad(v)))
    return V

### slower:
#def stiffness_vf(dim):
#    V = VForm(dim)
#    u, v = V.basisfuns()
#    V.add(inner(grad(u), grad(v)) * dx)
#    return V

def heat_st_vf(dim):
    V = VForm(dim, spacetime=True)
    u, v = V.basisfuns()
    V.add((inner(grad(u),grad(v)) + u.dt()*v) * dx)
    return V

def wave_st_vf(dim):
    V = VForm(dim, spacetime=True)
    u, v = V.basisfuns()
    utt_vt = u.dt(2) * v.dt()
    gradu_dtgradv = inner(grad(u), grad(v).dt())
    V.add((utt_vt + gradu_dtgradv) * dx)
    return V

def divdiv_vf(dim):
    V = VForm(dim, vec=dim**2)
    u, v = V.basisfuns(components=(dim,dim))
    V.add(div(u) * div(v) * dx)
    return V

