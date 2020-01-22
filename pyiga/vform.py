"""Functions and classes for representing and manipulating variational forms in an abstract way.
"""
from collections import OrderedDict, defaultdict
from functools import reduce
import operator
import numpy as np
import networkx
import copy
import numbers


def set_union(sets):
    return reduce(operator.or_, sets, set())

# Each AsmVar represents a named variable within the expression tree and has
# either an Expr (`expr`) or a source (`src`) determining how it is defined.
#
# If an expression is given, this means that the variable is defined as an
# expression in terms of other objects. In this case, the shape is determined
# automatically.
#
# Otherwise, the var has a src which is either
# - an InputField, which means that the variable is passed in as a function when
#   the assembler is created and evaluated in each needed quadrature point, or
# - a string prefixed with '@' which has a special meaning. Currently there is
#   only one such special var source:
#   '@gaussweights[i]', where i in range(0,d): the Gauss quadrature weight for
#       the i-th coordinate axis
#
# Note that basis functions (u, v) are not represented as AsmVars; instead,
# their expressions have the type PartialDerivExpr and store a reference to a
# BasisFun object.

class AsmVar:
    def __init__(self, name, src, shape, is_array=False, symmetric=False, deriv=None, depend_dims=None):
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
        self.is_array = is_array
        self.is_global = False  # is variable needed during assemble, or only setup?
        self.symmetric = (len(self.shape) == 2 and symmetric)
        self.deriv = deriv      # for input fields only
        self.depend_dims = depend_dims  # for input fields only (which axes does it depend on)
        self.as_expr = make_var_expr(self)

    def __str__(self):
        return self.name

    def is_scalar(self):
        return self.shape is ()
    def is_vector(self):
        return len(self.shape) == 1
    def is_matrix(self):
        return len(self.shape) == 2

class BasisFun:
    def __init__(self, name, vform, numcomp=None, component=None, space=0):
        self.name = name
        self.vform = vform
        self.numcomp = numcomp  # number of components; None means scalar
        self.component = component  # for vector-valued basis functions
        self.space = space
        self.asmgen = None  # to be set to AsmGenerator for code generation

class InputField:
    def __init__(self, name, shape, updatable=False):
        self.name = name
        self.shape = shape
        self.updatable = updatable

class VForm:
    """Abstract representation of a variational form."""
    def __init__(self, dim, arity=2, spacetime=False):
        self.dim = dim
        self.arity = arity
        self.vec = False
        self.spacetime = bool(spacetime)
        if self.spacetime:
            self.spacedims = range(self.dim - 1)
            self.timedim = self.dim - 1
        else:
            self.spacedims = range(self.dim)

        self.basis_funs = None
        self.inputs = []
        self.vars = OrderedDict()
        self.exprs = []         # expressions to be added to the result

        # predefined local variables with their generators (created on demand)
        self.predefined_vars = {
            'Jac':         lambda self: grad(self.Geo),
            'JacInv':      lambda self: inv(self.Jac),
            'GaussWeight': lambda self: self._gaussweight(),
            'W':           lambda self: self.GaussWeight * abs(det(self.Jac)),
        }
        # default input field: geometry transform
        self.Geo = self.input('geo', shape=(dim,))

    def _gaussweight(self):
        gw = [
            self.declare_sourced_var('gw%d' % i, shape=(), src='@gaussweights[%d]' % i,
                depend_dims=[i])
            for i in range(self.dim)
        ]
        return self.let('GaussWeight', reduce(operator.mul, gw))
        #return self.declare_sourced_var('GaussWeight', shape=(), src='@GaussWeight')

    def basisfuns(self, parametric=False, components=(None,None), spaces=(0,0)):
        """Obtain expressions representing the basis functions for this vform.

        Args:
            parametric (bool): by default, basis functions live in the physical domain
                (mapped by the geometry transform) and have their derivatives transformed
                accordingly. If `parametric=True` is given, they live in the parameter
                domain instead.
            components: for vector-valued problems, specify the number of components
                for each basis function here.
            spaces: space indices for problems where the basis functions live in
                different spaces.
        """
        if self.basis_funs is not None:
            raise RuntimeError('basis functions have already been constructed')

        def make_bfun_expr(bf):
            if bf.numcomp is not None:
                # return a vector which contains the components of the bfun
                vv = LiteralVectorExpr(
                    self.basisval(
                        BasisFun(bf.name, self, component=k),
                        physical=not parametric)
                    for k in range(bf.numcomp))
                return vv[0] if len(vv) == 1 else vv    # TODO: unify with scalar case?
            else:
                return self.basisval(bf, physical=not parametric)

        ar = self.arity
        # determine output size for vector assembler if needed
        if any(nc is not None for nc in components[:ar]):
            self.vec = reduce(operator.mul, components[:ar], 1)

        names = ('u', 'v')
        self.basis_funs = tuple(
                BasisFun(name, self, numcomp=nc, space=space)
                for (name,nc,space) in zip(names[:ar], components[:ar], spaces[:ar])
        )
        result = tuple(make_bfun_expr(bf) for bf in self.basis_funs)
        return result[0] if ar==1 else result

    def num_components(self):
        """Return number of vector components for each basis function space."""
        assert self.vec
        return tuple(bf.numcomp for bf in self.basis_funs)

    def input(self, name, shape=(), updatable=False):
        """Declare an input field with the given name and shape and return an
        expression representing it.

        If `updatable` is `True`, the generated assembler will allow updating of this
        field through an `update(name=value)` method.
        """
        inp = InputField(name, shape, updatable)
        self.inputs.append(inp)
        return self._input_as_varexpr(inp)

    def _input_as_varexpr(self, inp, deriv=0):
        """Return a VarExpr which refers to the given InputField with the desired derivative.

        This checks if a suitable variable has already been defined and if not, defines one.
        """
        # try to find existing AsmVar for this input/deriv combination
        inpvar = [v for v in self.vars.values() if v.src==inp and v.deriv==deriv]
        if len(inpvar) == 0:
            # no such var defined yet -- define it
            assert deriv <= 1, 'not implemented'
            deriv_tag = ('_grad') if deriv==1 else ''
            shape = inp.shape + ((self.dim,) if deriv==1 else ())
            varexpr = self.declare_sourced_var(inp.name + deriv_tag + '_a',
                    shape=shape, src=inp, deriv=deriv)
            varexpr.vf = self   # HACK to enable grad() to find the vf
            return varexpr
        elif len(inpvar) == 1:
            return inpvar[0].as_expr
        else:
            assert False, 'multiple variable definitions for input %s' % inp.name

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

    def set_var(self, name, var):
        if name in self.vars:
            raise KeyError('variable %s already declared' % name)
        self.vars[name] = var
        return var

    def let(self, name, expr, symmetric=False):
        """Define a variable with the given name which has the given expr as its value."""
        return self.set_var(name, AsmVar(name, expr, shape=None, symmetric=symmetric)).as_expr

    def declare_sourced_var(self, name, shape, src, symmetric=False, deriv=0, depend_dims=None):
        return self.set_var(name,
            AsmVar(name, src=src, shape=shape, is_array=True,
                symmetric=symmetric, deriv=deriv, depend_dims=depend_dims)).as_expr

    def add(self, expr):
        """Add an expression to this VForm."""
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
        # for each output component, replace one component of the basis functions
        # by the corresponding scalar basis function and all others by 0
        result = []

        if self.arity == 1:
            bfu = self.basis_funs[0]
            assert self.vec == bfu.numcomp, 'incorrect output size'
            for i in range(bfu.numcomp):
                expri = copy.deepcopy(expr)    # transform_expr is destructive, so copy the original
                expri = transform_expr(expri,
                    lambda e: self.replace_vector_bfuns(e, bfu.name, i),
                    type=PartialDerivExpr)
                result.append(expri)
            return as_vector(result)

        elif self.arity == 2:
            bfu, bfv = self.basis_funs
            assert self.vec == bfu.numcomp * bfv.numcomp, 'incorrect output size'

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
        else:
            assert False, 'invalid arity %d' % self.arity

    def basisval(self, basisfun, physical=False):
        return PartialDerivExpr(basisfun, self.dim * (0,), physical=physical)

    # automatically produce caching getters for predefined on-demand local variables
    def __getattr__(self, name):
        if name in self.vars:
            return self.vars[name].as_expr
        elif name in self.predefined_vars:
            expr = self.predefined_vars[name](self)
            if isinstance(expr, VarExpr):
                # no point in defining a new name for what is already a var
                return expr
            else:
                # define it as a new named variable and return it
                return self.let(name, expr)
        else:
            msg = "'{0}' object has no attribute '{1}'"
            raise AttributeError(msg.format(type(self).__name__, name))

    # expression analyis and transformations

    def dependency_graph(self):
        """Compute a directed graph of the dependencies between all used variables."""
        G = networkx.DiGraph()
        # add virtual basis function nodes to the graph
        G.add_nodes_from(self.basis_funs)

        for e in self.all_exprs(type=VarExpr):
            var = e.var
            G.add_node(var)
            if var.expr:
                for dep in var.expr.depends():
                    G.add_edge(dep, var)
        return G

    def transitive_deps(self, dep_graph, vars):
        """Return all vars on which the given vars depend directly or indirectly, in linearized order."""
        return set_union(networkx.ancestors(dep_graph, var) for var in vars)

    def linearize_vars(self, vars):
        """Returns an admissible order for computing the given vars, i.e., which
        respects the dependency relation."""
        return [var for var in self.linear_deps if var in vars]

    def transitive_closure(self, dep_graph, vars, exclude=set()):
        """Linearized transitive closure (w.r.t. dependency) of the given vars."""
        deps = set(vars) | self.transitive_deps(dep_graph, vars)
        return self.linearize_vars(deps - exclude)

    def vars_without_dep_on(self, dep_graph, exclude):
        """Return a linearized list of all expr vars which do not depend on the given vars."""
        nodes = set(dep_graph.nodes())
        # remove 'exlude' vars and anything that depends on them
        for var in exclude:
            nodes.discard(var)
            nodes -= networkx.descendants(dep_graph, var)
        return self.linearize_vars(nodes)

    def dependency_analysis(self, do_precompute=True):
        dep_graph = self.dependency_graph()
        self.linear_deps = list(networkx.topological_sort(dep_graph))

        # determine precomputable vars (no dependency on basis functions)
        precomputable = (self.vars_without_dep_on(dep_graph, self.basis_funs)
                if do_precompute else [])
        # only expression-based vars can be precomputed
        self.precomp = [v for v in precomputable if v.expr]
        # find deps of precomp vars which are pre-given (have src)
        pdeps = self.transitive_closure(dep_graph, self.precomp)
        self.precomp_deps = [v for v in pdeps if v.src]

        for var in self.precomp:
            # remove all dependencies since it's now precomputed
            # this ensures kernel_deps will not depend on dependencies of precomputed vars
            dep_graph.remove_edges_from(list(dep_graph.in_edges([var])))

        # compute linearized list of vars the kernel depends on
        kernel_deps = set_union(expr.depends() for expr in self.exprs)
        self.kernel_deps = self.transitive_closure(dep_graph, kernel_deps,
                exclude=set(self.basis_funs))

        for var in self.kernel_deps:
            # ensure precomputed kernel deps get array storage
            if var in self.precomp:
                var.is_array = True
            # make arrays for kernel dependencies global (store as class member)
            if var.is_array:
                var.is_global = True

        # separate precomp into locals (not used in kernel, only during precompute) and true dependencies
        self.precomp_locals = [v for v in self.precomp if not v.is_global]
        self.precomp        = [v for v in self.precomp if v.is_global]

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

    def finalize(self, do_precompute=True):
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
        self.dependency_analysis(do_precompute=do_precompute)

    def find_max_deriv(self):
        #return max((max(e.D) for e in self.all_exprs(type=PartialDerivExpr)), default=0)
        # Py2.7
        try:
            return max((max(e.D) for e in self.all_exprs(type=PartialDerivExpr)))
        except ValueError:  # empty iterator
            return 0


################################################################################
# Expressions for use in variational forms
################################################################################

class Expr:
    """Abstract base class which all expressions derive from.

    Attributes:
        shape: the shape of the expression as a tuple, analogous to
            :attr:`numpy.ndarray.shape`.
            Scalar expressions have the empty tuple ``()`` as their shape.
    """
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
    def __abs__(self):  return BuiltinFuncExpr('abs', self)

    def __bool__(self):  return True
    __nonzero__ = __bool__  # Python 2 compatibility

    def __len__(self):
        if self.is_scalar():
            raise TypeError('cannot get length of scalar')
        else:
            return self.shape[0]

    # convenience accessors for child nodes
    @property
    def x(self):
        """Return the first child expression."""
        return self.children[0]
    @property
    def y(self):
        """Return the second child expression."""
        return self.children[1]
    @property
    def z(self):
        """Return the third child expression."""
        return self.children[2]

    def is_scalar(self):
        """Returns True iff the expression is scalar."""
        return self.shape is ()
    def is_vector(self):
        """Returns True iff the expression is vector-valued."""
        return len(self.shape) == 1
    def is_matrix(self):
        """Returns True iff the expression is matrix-valued."""
        return len(self.shape) == 2

    def depends(self):
        return set_union(x.depends() for x in self.children)

    def is_var_expr(self):
        return False

    def dx(self, k, times=1):
        """Compute a partial derivative. Equivalent to :func:`Dx`."""
        return Dx(self, k, times)

    def dt(self, times=1):
        return Dt(self, times)

    def dot(self, x):
        """Returns the dot product of this with `x`; see :func:`dot` for semantics."""
        return dot(self, x)
    __matmul__ = dot        # allow @ operator in Python 3.5+

    @property
    def T(self):
        """For a matrix expression, return its transpose."""
        if not self.is_matrix():
            raise TypeError('can only transpose matrices')
        return TransposedMatrixExpr(self)

    def ravel(self):
        """Ravel a matrix expression into a vector expression."""
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
        if self.children:
            new = copy.copy(self)
            new.children = copy.deepcopy(self.children, memo)
            return new
        else:
            # we only care about reproducing the tree structure --
            # make sure we don't clone VarExprs etc which depend on identity
            return self

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
    def hash_key(self):
        return (self.value,)
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
    def __str__(self):
        return '(' + ',\n '.join(str(self[i,:]) for i in range(self.shape[0])) + ')'
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

class BuiltinFuncExpr(Expr):
    def __init__(self, funcname, expr):
        expr = as_expr(expr)
        if not expr.is_scalar(): raise TypeError('can only compute %s of scalars' % funcname)
        self.funcname = funcname
        self.shape = ()
        self.children = (expr,)
    def __str__(self):
        return 'abs(%s)' % str(self.x)
    def gencode(self):
        f = self.func_to_code.get(self.funcname, self.funcname)
        return '%s(%s)' % (f, self.x.gencode())
    func_to_code = {
            'abs' : 'fabs',
    }

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
        return set((self.basisfun,))

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
            # Py2.7
            #yield from recurse(c)
            for x in recurse(c): yield x
        if (deep and e.is_var_expr()
                 and e.var.expr is not None):
            # Py2.7
            #yield from recurse(e.var.expr)
            for x in recurse(e.var.expr): yield x
        if type is None or isinstance(e, type):
            yield e
    for e in exprs:
        # Py2.7
        #yield from recurse(e)
        for x in recurse(e): yield x

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

#: A symbolic expression representing the integration weight stemming from the geometry map.
dx = VolumeMeasureExpr()

def Dx(expr, k, times=1):
    """Partial derivative of `expr` along the `k`-th coordinate axis."""
    if expr.is_var_expr() and expr.var.expr:
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
    if expr.is_var_expr() and expr.var.expr:
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
    """Gradient of an expression.

    If `expr` is scalar, results in a vector of all partial derivatives.

    If `expr` is a vector, results in the Jacobian matrix whose rows are
    the gradients of the individual components.

    If `dims` is specified, it is a tuple of dimensions along which to take
    the derivative. By default, all space dimensions are used.
    """
    if expr.is_var_expr():
        if expr.var.expr:
            expr = expr.var.expr    # access underlying expression - mild hack
        if expr.var.src:
            # gradient/Jacobian of an input field
            s = expr.var.src
            assert isinstance(s, InputField), 'can only compute gradients of input fields'
            assert dims is None, 'can only compute full gradient'
            return expr.vf._input_as_varexpr(s, deriv=expr.var.deriv+1)
    if expr.is_vector():
        return as_matrix([grad(z, dims=dims) for z in expr])  # compute Jacobian of vector expression
    if not isinstance(expr, PartialDerivExpr):
        raise TypeError('can only compute gradient of basis function')
    if dims is None:
        dims = expr.basisfun.vform.spacedims
    return LiteralVectorExpr(Dx(expr, k) for k in dims)

def div(expr):
    """The divergence of a vector-valued expressions, resulting in a scalar."""
    if not expr.is_vector():
        raise TypeError('can only compute divergence of vector expression')
    return tr(grad(expr))

def curl(expr):
    """The curl (or rot) of a 3D vector expression."""
    if not (expr.is_vector() and len(expr) == 3):
        raise TypeError('can only compute curl of 3D vector expression')
    return as_vector((
        expr.z.dx(1) - expr.y.dx(2),
        expr.x.dx(2) - expr.z.dx(0),
        expr.y.dx(0) - expr.x.dx(1),
    ))

def as_expr(x):
    """Interpret input as an expression; useful for constants."""
    if isinstance(x, Expr):
        return x
    elif isinstance(x, numbers.Real):
        return ConstExpr(x)
    else:
        raise TypeError('cannot coerce %s to expression' % x)

def as_vector(x):
    """Convert a sequence of expressions to a vector expression."""
    return LiteralVectorExpr(x)
def as_matrix(x):
    """Convert a sequence of sequence of expressions to a matrix expression."""
    return LiteralMatrixExpr(x)

def inner(x, y):
    """The inner product of two vector or matrix expressions."""
    if not (x.is_vector() or x.is_matrix()):
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
    """The dot product of two expressions.

    Depending on the shapes of the arguments `a` and `b`, its semantics differ:

    * vector, vector: inner product (see :func:`inner`)
    * matrix, vector: matrix-vector product
    * matrix, matrix: matrix-matrix product
    """
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
    """Cross product of two 3D vectors."""
    return VectorCrossExpr(x, y)

def outer(x, y):
    """Outer product of two vectors, resulting in a matrix."""
    return OuterProdExpr(x, y)

def sqrt(x):
    """Square root of an expression."""
    return BuiltinFuncExpr('sqrt', x)
def exp(x):
    """Exponential of an expression."""
    return BuiltinFuncExpr('exp', x)
def log(x):
    """Natural logarithm of an expression."""
    return BuiltinFuncExpr('log', x)
def sin(x):
    """Sine of an expression."""
    return BuiltinFuncExpr('sin', x)
def cos(x):
    """Cosine of an expression."""
    return BuiltinFuncExpr('cos', x)
def tan(x):
    """Tangent of an expression."""
    return BuiltinFuncExpr('tan', x)

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
    V = VForm(dim)
    u, v = V.basisfuns(components=(dim,dim))
    V.add(div(u) * div(v) * dx)
    return V

def L2functional_vf(dim):
    V = VForm(dim, arity=1)
    u = V.basisfuns()
    f = V.input('f', shape=(), updatable=True)
    V.add(f * u * dx)
    return V
