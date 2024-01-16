"""Functions and classes for representing and manipulating variational forms in an abstract way.

For a detailed user guide, see :doc:`/guide/vforms`.
"""
from collections import OrderedDict, defaultdict
from functools import reduce
from enum import IntEnum
import operator
import numpy as np
import networkx
import copy
import numbers

def set_union(sets):
    return reduce(operator.or_, sets, set())

def _D_to_indices(D):
    """For a derivative tuple D, return the corresponding indices of the derivatives."""
    D = list(D)
    indices = []
    while sum(D) > 0:
        i = np.flatnonzero(D)[0]
        D[i] -= 1
        indices.append(i)
    return tuple(indices)

def sym_index_to_seq(n, i, j):
    """Convert index (i,j) into a n x n symmetric matrix into a sequential
    index which lies in `range(0, n * (n + 1) // 2)`."""
    if i > j:           # make sure j >= i
        i, j = j, i
    idx = sum(n - k for k in range(0, i))   # diagonal index on row i
    return idx + (j - i)

def _integer_power(x, y):
    if y < 0:
        return 1.0 / _integer_power(x, -y)
    elif y == 0:
        return 1.0
    elif y == 1:
        return x
    else:
        return _integer_power(x, y-1) * x

def _jac_to_unscaled_normal(jac):
    if jac.shape == (2, 1):     # line integral
        x = jac[:, 0]
        return as_vector((-x[1], x[0]))
    elif jac.shape == (3, 2):   # surface integral
        x, y = jac[:, 0], jac[:, 1]
        return cross(x, y)
    else:
        assert False, 'do not know how to compute normal vector for Jacobian shape {}'.format(jac.shape)

class Scope(IntEnum):
    """An enum describing what an expr or var depends on."""
    CONSTANT = 0        # globally constant
    FIELD    = 1        # changes per quadrature node, but does not depend on basis functions
    BASISFUN = 2        # depends on basis functions

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
# - a Parameter, which means that the variable is passed as a constant value
#   before assembling.
#
# Note that basis functions (u, v) are not represented as AsmVars; instead,
# their expressions have the type PartialDerivExpr and store a reference to a
# BasisFun object.

class AsmVar:
    def __init__(self, vf, name, src, shape, symmetric=False, deriv=None):
        self.vform = vf
        self.name = name
        if isinstance(src, Expr):
            self.expr = src
            self.shape = src.shape
            self.src = None
            self.scope = self.expr.scope()
        else:
            self.src = src
            self.expr = None
            assert shape is not None
            self.shape = shape
            self.scope = self.src.scope
        self.is_global = False  # is variable needed during assemble, or only setup?
        self.symmetric = (len(self.shape) == 2 and symmetric)
        self.deriv = deriv      # for input fields only
        self.as_expr = make_var_expr(vf, self)

    def hash(self, expr_hashes):
        # helper for VForm.hash()
        src_hash = None
        if self.expr:
            src_hash = expr_hashes[self.expr]
        elif isinstance(self.src, InputField) or isinstance(self.src, Parameter):
            src_hash = self.src.hash()
        elif isinstance(self.src, str):
            src_hash = hash(self.src)
        else:
            assert False, 'no expr and invalid src'
        return hash((self.name, src_hash, self.shape, self.symmetric, self.deriv))

    def __str__(self):
        return self.name

    def is_scalar(self):
        return self.shape == ()
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
        self.scope = Scope.BASISFUN
    def hash(self):
        return hash((self.name, self.numcomp, self.component, self.space))
    def __str__(self):
        s = 'bfun(%s' % self.name
        if self.component is not None:
            s += '[%s]' % self.component
        return s + ')'

class InputField:
    """A function defined in parametric or physical coordinates which is passed
    as an input when assembling."""
    def __init__(self, name, shape, physical, vform, updatable=False):
        self.name = name
        self.shape = shape
        self.physical = physical
        self.vform = vform
        self.updatable = updatable
        self.scope = Scope.FIELD
    def hash(self):
        return hash((self.name, self.shape, self.physical, self.updatable))

class Parameter:
    """A constant value (scalar, vector, matrix...) which is passed as an input
    when assembling."""
    def __init__(self, name, shape):
        self.name = name
        self.shape = shape
        self.scope = Scope.CONSTANT
        self.symmetric = False          # currently only used for storage_size()
    def hash(self):
        return hash((self.name, self.shape))

class VForm:
    """Abstract representation of a variational form.

    See :doc:`/guide/vforms` for a comprehensive user guide.

    Args:
        dim (int): the space dimension
        geo_dim (int): the dimension of the image of the geometry map. By
            default, it is equal to `dim`. It should be either `dim` (for
            volume integrals) or `dim + 1` (for surface integrals).
        arity (int): the arity of the variational form, i.e., 1 for a linear
            functional and 2 for a bilinear form
        spacetime (bool): whether the form describes a space-time discretization (deprecated)
    """
    def __init__(self, dim, geo_dim=None, boundary=False, arity=2, spacetime=False):
        self.dim = dim
        if geo_dim is None:
            geo_dim = dim
        self.arity = arity
        self.geo_dim = geo_dim
        self.is_boundary = bool(boundary)
        self.vec = False
        self.spacetime = bool(spacetime)
        if self.spacetime:
            self.spacedims = range(self.dim - 1)
            self.timedim = self.dim - 1
        else:
            self.spacedims = range(self.dim)

        self.basis_funs = None
        self.inputs = []
        self.params = []
        self.vars = OrderedDict()
        self.exprs = []         # expressions to be added to the result

        def _volume_weight(self):
            if not self.is_volume_integral():
                raise ValueError('volume measure not defined for surface integral')
            return self.GaussWeight * abs(det(self.Jac))

        def _surface_weight(self):
            if self.is_volume_integral():
                raise ValueError('surface measure not defined for volume integral')
            return self.GaussWeight * norm(_jac_to_unscaled_normal(self.BJac))

        def _surface_normal(self):
            if self.is_volume_integral():
                raise ValueError('surface measure not defined for volume integral')
            un = _jac_to_unscaled_normal(self.BJac)
            return un / norm(un)

        def _Jac_to_boundary(self):
            """A constant matrix which reduces the Jacobian to only the boundary part."""
            if not self.is_boundary_integral():
                raise ValueError('_Jac_to_boundary only defined for boundary integrals')
            return self.parameter('Jac_to_boundary', (self.dim, self.dim - 1))

        def _BJac(self):
            """Boundary Jacobian (shape (k+1) x k)."""
            if self.is_surface_integral():
                return self.Jac
            elif self.is_boundary_integral():
                return self.Jac @ self.Jac_to_boundary
            else:
                raise ValueError('BJac not defined for volume integrals')

        # predefined local variables with their generators (created on demand)
        self.predefined_vars = {
            # for spacetime, Jac still refers to the full d+1 Jacobian
            'Jac':         lambda self: grad(self.Geo, dims=range(self.dim) if self.spacetime else None, parametric=True),
            'JacInv':      lambda self: inv(self.Jac),
            'GaussWeight': lambda self: self._gaussweight(),
            'W':           _volume_weight,
            'SW':          _surface_weight,
            'normal':      _surface_normal,
            'Jac_to_boundary': _Jac_to_boundary,
            'BJac':         _BJac,
        }
        # default input field: geometry transform
        self.Geo = self.input('geo', shape=(geo_dim,))
        self.__hash = None
        self.__is_finalized = False

    def is_volume_integral(self):
        return self.dim == self.geo_dim and not self.is_boundary

    def is_surface_integral(self):
        return self.dim == self.geo_dim - 1 and not self.is_boundary

    def is_boundary_integral(self):
        return self.is_boundary

    def hash(self):
        # A hash to avoid recompiling the same vform over and over again.
        # NB: 1. Transforming the vform, such as during compilation by
        # finalize(), changes the exprs and therefore the hash.
        # 2. This does not persist across processes since hash() is salted.
        # However, compilation of the generated source code is cached separately.
        if self.__hash is None:
            # vforms are considered immutable after initial setup - compute only once
            expr_hashes = self.compute_recursive(lambda e, child_hashes: e.hash(child_hashes))
            self.__hash = hash((self.dim, self.arity, self.vec, self.spacetime) +
                    tuple(bf.hash() for bf in self.basis_funs) +
                    tuple(inp.hash() for inp in self.inputs) +
                    tuple(var.hash(expr_hashes) for var in self.vars.values()) +
                    tuple(expr_hashes[e] for e in self.exprs))
        return self.__hash

    def _gaussweight(self):
        gw = [GaussWeightExpr(i) for i in range(self.dim)]
        return reduce(operator.mul, gw)

    def basisfuns(self, components=(None,None), spaces=(0,0)):
        """Obtain expressions representing the basis functions for this vform.

        Args:
            components: for vector-valued problems, specify the number of components
                for each basis function here.
            spaces: space indices for problems where the basis functions live in
                different spaces.
        """
        if self.basis_funs is not None:
            raise RuntimeError('basis functions have already been constructed')

        def make_bfun_expr(bf):
            derivs = self.dim * (0,)
            if bf.numcomp is not None:
                # return a vector which contains the components of the bfun
                vv = LiteralVectorExpr(
                    PartialDerivExpr(
                        BasisFun(bf.name, self, component=k),
                        derivs)
                    for k in range(bf.numcomp))
                return vv[0] if len(vv) == 1 else vv    # TODO: unify with scalar case?
            else:
                return PartialDerivExpr(bf, derivs)

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

    def num_spaces(self):
        """Return the number of different function spaces used in this VForm."""
        return len(np.unique(tuple(bf.space for bf in self.basis_funs)))

    def input(self, name, shape=(), physical=False, updatable=False):
        """Declare an input field with the given name and shape and return an
        expression representing it.

        By default, input fields are assumed to be given in parametric
        coordinates. If the field is defined in physical geometry coordinates,
        pass `physical=True`.

        If `updatable` is `True`, the generated assembler will allow updating of this
        field through an `update(name=value)` method.
        """
        inp = InputField(name, shape, physical, self, updatable)
        self.inputs.append(inp)
        return self._input_as_expr(inp)

    def parameter(self, name, shape=()):
        """Declare a named constant parameter with the given shape and return
        an expression representing it.
        """
        param = Parameter(name, shape)
        self.params.append(param)
        return self.declare_sourced_var(name, shape, param)

    def _input_as_expr(self, inp, deriv=0):
        """Return an expr which refers to the given InputField with the desired derivative.

        This checks if a suitable variable has already been defined and if not, defines one.
        """
        # try to find existing AsmVar for this input/deriv combination
        inpvar = [v for v in self.vars.values() if v.src==inp and v.deriv==deriv]
        if len(inpvar) == 0:
            # no such var defined yet -- define it
            assert 0 <= deriv <= 2, 'not implemented'
            if deriv == 2 and len(inp.shape) > 1:
                raise RuntimeError('Hessian only implemented for scalar and vector input functions')
            deriv_tag = ['', '_grad', '_hess'][deriv]
            d = self.dim
            deriv_dim = [(), (d,), ((d+1)*d//2,)][deriv]        # Hessian stores the symmetric part only
            shape = inp.shape + deriv_dim
            varexpr = self.declare_sourced_var(inp.name + deriv_tag + '_a',
                    shape=shape, src=inp, deriv=deriv)
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

    def pderiv_as_var(self, bfun, indices=None, D=None):
        if D is None:
            D = self.indices_to_D(indices)
        name = '_d%s_%s' % (bfun.name, ''.join(str(k) for k in D))
        if not name in self.vars:
            self.let(name, PartialDerivExpr(bfun, D, physical=False))
        return self.vars[name].as_expr

    def set_var(self, name, var):
        if name in self.vars:
            raise KeyError('variable %s already declared' % name)
        self.vars[name] = var
        return var

    def let(self, name, expr, symmetric=False):
        """Define a variable with the given name which has the given expr as its value."""
        return self.set_var(name, AsmVar(self, name, expr, shape=None, symmetric=symmetric)).as_expr

    def declare_sourced_var(self, name, shape, src, symmetric=False, deriv=0):
        return self.set_var(name,
            AsmVar(self, name, src=src, shape=shape, symmetric=symmetric, deriv=deriv)).as_expr

    def add(self, expr):
        """Add an expression to this VForm."""
        if self.__hash is not None:
            raise RuntimeError('can no longer modify this VForm')
        if not expr.is_scalar():
            raise TypeError('all expressions added to a VForm must be scalar')
        if self.vec:
            expr = self.substitute_vec_components(expr)
            if expr.is_matrix():
                expr = expr.ravel()
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

    # automatically produce caching getters for predefined on-demand local variables
    def __getattr__(self, name):
        if name in self.vars:
            return self.vars[name].as_expr
        elif name in self.predefined_vars:
            expr = self.predefined_vars[name](self)
            if name in self.vars:
                # the predefined_vars function may have already registered it
                assert self.vars[name].as_expr is expr
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

        for e in self.all_exprs(type=VarRefExpr):
            var = e.var
            G.add_node(var)
            if var.expr:
                for dep in var.expr.depends():
                    G.add_edge(dep, var)
        return G

    def transitive_deps(self, dep_graph, vars):
        """Return the set of all vars on which the given vars depend directly or indirectly."""
        return set_union(networkx.ancestors(dep_graph, var) for var in vars)

    def linearize_vars(self, vars):
        """Returns an admissible order for computing the given vars, i.e., which
        respects the dependency relation."""
        return [var for var in self.linear_deps if var in vars]

    def transitive_closure(self, dep_graph, vars, exclude=set()):
        """Linearized transitive closure (w.r.t. dependency) of the given vars."""
        deps = set(vars) | self.transitive_deps(dep_graph, vars)
        return self.linearize_vars(deps - exclude)

    def dependency_analysis(self, do_precompute=True):
        dep_graph = self.dependency_graph()
        self.linear_deps = list(networkx.topological_sort(dep_graph))

        # determine precomputable vars (no dependency on basis functions)
        if do_precompute:
            self.precomp = [v for v in self.linear_deps if v.scope != Scope.BASISFUN]
        else:
            self.precomp = []

        for var in self.precomp:
            # remove all dependencies since it's now precomputed
            # this ensures kernel_deps will not depend on dependencies of precomputed vars
            dep_graph.remove_edges_from(list(dep_graph.in_edges([var])))

        # compute linearized list of vars (excluding basis functions) the kernel depends on
        kernel_deps = set_union(expr.depends() for expr in self.exprs)
        self.kernel_deps = self.transitive_closure(dep_graph, kernel_deps,
                exclude=set(self.basis_funs))

        for var in self.kernel_deps:
            # globals are either input fields the kernel depends on or variables
            # which are computed in the precompute function
            if var in self.precomp or isinstance(var.src, InputField):
                var.is_global = True

    def all_exprs(self, type=None, once=True):
        """Deep, depth-first iteration of all expressions with dependencies.

        If `type` is given, only exprs which are instances of that type are yielded.
        If `once=True` (default), each expr is visited only once.
        """
        return iterexprs(self.exprs, deep=True, type=type, once=once)

    def transform(self, fun, type=None, deep=True):
        """Apply `fun` to all exprs (or all exprs of the given `type`). If `fun` returns
        an expr, replace the old expr by this new one.
        """
        self.exprs = transform_exprs(self.exprs, fun, type=type, deep=deep)

    def collect(self, type=None, filter=None):
        for e in self.all_exprs(type=type):
            if filter is None or filter(e):
                yield e

    def replace_physical_derivs(self, e):
        if sum(e.D) == 0:       # no derivatives?
            return e.make_parametric()

        dst_phys = e.is_physical_deriv()
        if isinstance(e, VarRefExpr):
            src_phys = e.var.src.physical
        else:
            # derivative of input function - always defined in parametric coordinates
            src_phys = False

        if (not src_phys) and (not dst_phys):
            return  # parametric derivative of parametric variable - no transformation
        if src_phys and dst_phys:
            return  # physical derivative of physical field - no transformation
        if src_phys and (not dst_phys):
            raise RuntimeError('cannot compute parametric derivative of physical input field')

        # only remaining case: source parametric with physical derivative

        if self.spacetime:
            # the following assumes a space-time cylinder -- can keep time derivatives parametric,
            # only need to transform space derivatives
            assert self.timedim == self.dim - 1 # to make sure the next line is correct
            D_x = e.D[:-1] + (0,)   # HACK: should be D[spacedims]; assume time is last
            if sum(D_x) == 0:
                return self.pderiv_as_var(e.basisfun, D=e.D)   # time derivatives are parametric
            elif sum(D_x) == 1:
                k = D_x.index(1)
                dts = e.D[-1] * (self.timedim,)
                spacegrad = as_vector(self.pderiv_as_var(e.basisfun, indices=(i,) + dts)
                                      for i in self.spacedims)
                return inner(self.JacInv[self.spacedims, k], spacegrad)
        else:
            order = sum(e.D)
            if order == 1:
                (k,) = _D_to_indices(e.D)    # get index of derivative direction
                assert e.is_scalar()
                return inner(self.JacInv[:, k], grad(e.without_derivs(), parametric=True))
            elif order == 2:
                i,j = _D_to_indices(e.D)
                assert e.is_scalar()
                Hp = hess(e.without_derivs(), parametric=True)
                gp = grad(e.without_derivs(), parametric=True)

                # transform the parametric Hessian
                # H_ij = self.JacInv.T.dot(Hp.dot(self.JacInv))[i,j]    # equivalent:
                H_ij = self.JacInv[:,i].dot(Hp.dot(self.JacInv[:,j]))
                # add the contributions from the geometry Hessian
                for k in range(self.dim):
                    H_ij = H_ij + gp[k] * self._geo_hess_trf(k, i, j)
                return H_ij

        assert False, 'higher order physical derivatives not implemented'

    def _geo_hess_trf(self, a, i, j):
        # implements formula (A.12) for the (i,j)-th entry of the physical
        # Hessian of the a-th component of the inverse geometry transform from:
        #   L. Dalcin, N. Collier, P. Vignal, A.M.A. Côrtes, V.M. Calo:
        #   PetIGA: A framework for high-performance isogeometric analysis,
        #   https://doi.org/10.1016/j.cma.2016.05.011.
        # BUT NOTE: there is a sign error in the paper!
        var_name = '_geo_hess_trf_{}_{}_{}'.format(a, i, j)
        if var_name in self.vars:
            return self.vars[var_name].as_expr
        else:
            d = self.dim
            J = self.JacInv
            expr = -sum(hess(self.Geo[m], parametric=True)[e,u] * J[a,m] * J[e,i] * J[u,j]
                        for m in range(d) for e in range(d) for u in range(d))
            return self.let(var_name, expr)

    def insert_input_field_derivs(self, e):
        if sum(e.D) == 0:
            return
        assert e.is_input_var_expr()    # only those can have nonzero derivatives

        # either parametric derivatives of parametric input field,
        # or physical derivative of physical input field
        assert bool(e.parametric) == (not e.var.src.physical)

        assert e.var.deriv == 0     # we always refer to the base var without derivs

        if sum(e.D) == 1:
            G = e._para_grad()
            (k,) = _D_to_indices(e.D)
            return G[k]
        elif sum(e.D) == 2:
            H = e._para_hess()
            i,j = _D_to_indices(e.D)
            return H[sym_index_to_seq(len(e.D), i, j)]
        else:
            assert False, 'higher order physical derivatives not implemented'

    def para_derivs_to_vars(self, e):
        if not e.physical and sum(e.D) > 0:
            return self.pderiv_as_var(e.basisfun, D=e.D)

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
            used_vars = set(e.var for e in self.collect(type=VarRefExpr, filter=lambda e: e.var.expr))
            all_root_exprs = [var.expr for var in used_vars] + list(self.exprs)

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
                    if expr_count[h] > 1 and complexity[e] > 2]
            if not cse:
                break

            # find the most complex among all common subexprs
            biggest_cse = max(cse, key=complexity.get)
            h = hashes[biggest_cse]

            # extract it into a new variable
            var = self.let(tmpname(), biggest_cse)
            self.transform(lambda e: var if hashes[e] == h else None)
        return tmpidx[0] > 0    # did we do anything?

    def replace_trivial_vars(self, e):
        if e.var.expr:
            inner_expr = e.get_underlying_expr()
            if isinstance(inner_expr, VarRefExpr):
                return inner_expr

    def finalize(self, do_precompute=True):
        """Performs standard transforms and dependency analysis."""
        if self.__is_finalized:
            raise RuntimeError('VForm has already been finalized')
        # make sure the hash is computed on the initial expression tree
        self.hash()
        # replace "dx" and "ds" by quadrature weight function
        self.transform(lambda e: self.W, type=VolumeMeasureExpr)
        self.transform(lambda e: self.SW, type=SurfaceMeasureExpr)
        # replace physical derivs by proper expressions in terms of parametric derivs
        self.transform(self.replace_physical_derivs, type=PartialDerivExpr)
        self.transform(self.replace_physical_derivs, type=VarRefExpr)
        # replace derivatives of input fields by the proper array variables
        self.transform(self.insert_input_field_derivs, type=VarRefExpr)
        # replace parametric derivs by named vars (for readability only)
        self.transform(self.para_derivs_to_vars, type=PartialDerivExpr, deep=False)
        # convert all expressions to scalar form
        self.transform(_to_literal_vec_mat)
        # fold constants, eliminate zeros
        self.transform(lambda e: e.fold_constants())
        # find common subexpressions and extract them into named variables
        self.extract_common_expressions()
        # eliminate variables which directly refer to other variables
        self.transform(self.replace_trivial_vars, type=VarRefExpr)
        # perform dependency analysis for expressions and variables
        self.dependency_analysis(do_precompute=do_precompute)
        self.__is_finalized = True

    def find_max_deriv(self):
        return max((max(e.D) for e in self.all_exprs(type=PartialDerivExpr)), default=0)


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

    def __pow__(self, z):
        if not self.is_scalar():
            raise TypeError('cannot take power of non-scalar expression')
        if isinstance(z, ConstExpr):
            y = int(z.value)
            if y != z.value:
                raise TypeError('only integer powers implemented')
            z = y
        if not isinstance(z, numbers.Integral):
            raise TypeError('only integer powers implemented')
        return as_expr(_integer_power(self, z))

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
        return self.shape == ()
    def is_vector(self):
        """Returns True iff the expression is vector-valued."""
        return len(self.shape) == 1
    def is_matrix(self):
        """Returns True iff the expression is matrix-valued."""
        return len(self.shape) == 2

    def depends(self):
        return set_union(x.depends() for x in self.children)

    def scope(self):
        return max(c.scope() for c in self.children)

    def is_var_expr(self):
        return False
    def is_input_var_expr(self):
        return False

    def dx(self, k, times=1, parametric=False):
        """Compute a partial derivative. Equivalent to :func:`Dx`."""
        return Dx(self, k, times, parametric=parametric)

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
        return LiteralMatrixExpr(
                [[self[j,i] for j in range(self.shape[0])]
                            for i in range(self.shape[1])])

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
        """A tuple of values which uniquely identifies an instance of this expression."""
        return ()

    def hash(self, child_hashes):
        return hash((type(self), self.shape) + self.hash_key() + child_hashes)

    def is_constant(self, val):
        return False

    def is_zero(self):
        return self.is_constant(0)

    def fold_constants(self):
        return self

    def find_vf(self):
        """Attempt to determine the ambient VForm from the expression tree."""
        for c in self.children:
            vf = c.find_vf()
            if vf:
                return vf

    def __deepcopy__(self, memo):
        if self.children:
            new = copy.copy(self)
            new.children = copy.deepcopy(self.children, memo)
            return new
        else:
            # we only care about reproducing the tree structure --
            # make sure we don't clone exprs which depend on identity
            # (are any such left? VarExpr no longer exists)
            return self

    base_complexity = 1

def make_var_expr(vf, var):
    """Create an expression of the proper shape which refers to the variable `var`."""
    shape = var.shape
    if shape == ():
        return VarRefExpr(var, ())
    elif len(shape) == 1:
        return LiteralVectorExpr(
                [VarRefExpr(var, (i,)) for i in range(shape[0])])
    elif len(shape) == 2:
        return LiteralMatrixExpr(
                [[VarRefExpr(var, (i,j)) for j in range(shape[1])]
                                             for i in range(shape[0])])
    else:
        assert False, 'invalid shape'

class ConstExpr(Expr):
    """A constant scalar value."""
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
    def _dx_impl(self, k, times, parametric):
        return as_expr(0) if times > 0 else self
    def scope(self):
        return Scope.CONSTANT
    base_complexity = 0

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
        if isinstance(entries, Expr):
            if not entries.is_matrix():
                raise TypeError('expression is not a matrix')
            entries = [[entries[i,j] for j in range(0, entries.shape[1])]
                       for i in range(0, entries.shape[0])]
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

class VarRefExpr(Expr):
    """A scalar expression which refers to an entry of a variable or a derivative thereof."""
    def __init__(self, var, I, D=None, parametric=False):
        assert isinstance(var, AsmVar)
        I = tuple(I)
        assert len(I) == len(var.shape)
        self.shape = ()
        self.var = var
        self.I = I
        self.children = ()
        if D is None:
            D = self.var.vform.dim * (0,)
        self.D = tuple(D)
        assert sum(self.D) == 0 or self.is_input_var_expr(),\
                'derivatives only valid for input vars'
        self.parametric = parametric

    def depends(self):
        return set((self.var,))

    def scope(self):
        return self.var.scope

    def __str__(self):
        if len(self.I) > 0:
            join_I = ','.join(str(i) for i in self.I)
            s = '%s[%s]' % (self.var.name, join_I)
        else:
            s = self.var.name
        if sum(self.D) != 0:
            s += '_' + ''.join(str(k) for k in self.D)
            s += '(para)' if self.parametric else '(phys)'
        return s

    def hash_key(self):
        return (self.var.name, self.I, self.D, self.parametric)

    def to_seq(self):
        if len(self.I) == 1:
            return self.I[0]
        else:
            i, j = self.I
            if self.var.symmetric and i > j:
                i, j = j, i
            return i * self.var.shape[1] + j

    def is_var_expr(self):
        return True
    def is_input_var_expr(self):
        return isinstance(self.var.src, InputField)

    def _para_grad(self):
        # generate a new expr for the parametric gradient (input vars only)
        assert self.is_input_var_expr(), '_para_grad only handles input fields'
        if len(self.I) == 0:
            return self.var.vform._input_as_expr(self.var.src, deriv=self.var.deriv+1)
        elif len(self.I) == 1:
            return self.var.vform._input_as_expr(self.var.src, deriv=self.var.deriv+1)[self.I[0], :]
        else:
            assert False, 'gradient of matrices not implemented'

    def _para_hess(self):
        # generate a new expr for the parametric Hessian matrix (input vars only)
        assert self.is_input_var_expr(), '_para_hess only handles input fields'
        assert self.is_scalar(), 'can only compute Hessian of scalars'
        assert self.var.deriv == 0
        assert sum(self.D) == 2
        H = self.var.vform._input_as_expr(self.var.src, deriv=2)
        if self.I == ():
            return H
        elif len(self.I) == 1:
            return H[self.I[0], :]      # Hessian components corresponding to the i-th component
        else:
            raise RuntimeError('Hessian only implemented for scalar and vector input variables')

    def is_physical_deriv(self):
        return (not self.parametric)

    def make_parametric(self):
        if self.parametric:
            return self
        else:
            return VarRefExpr(self.var, self.I, self.D, parametric=True)

    def without_derivs(self):
        """Return a reference to the underlying variable without derivatives."""
        return VarRefExpr(self.var, self.I, len(self.D) * (0,), parametric=True)

    def get_underlying_expr(self):
        # for a var defined using an expr, get the corresponding scalar expr
        assert self.var.expr, 'only valid for expr-based vars'
        if self.I == ():
            return self.var.expr
        elif len(self.I) == 1:
            return self.var.expr[self.I[0]]
        else:
            return self.var.expr[self.I]

    def _dx_impl(self, k, times, parametric):
        if not (parametric == self.parametric or sum(self.D) == 0):
            raise RuntimeError('cannot mix physical and parametric derivatives')
        if times == 0:
            return self
        if self.is_input_var_expr():
            # For input vars, simply create a symbolic reference to the partial
            # derivative.  Actual derivatives will be substituted in a
            # transformation step.
            Dnew = list(self.D)
            Dnew[k] += times
            return VarRefExpr(self.var, self.I, Dnew, parametric)
        elif isinstance(self.var.src, Parameter):
            # parameters are constants, so any derivatives are 0
            return ConstExpr(0)
        elif self.var.expr:
            # in case of a variable, compute derivative of the underlying expression
            return Dx(self.get_underlying_expr(), k, times, parametric=parametric)
        else:
            raise TypeError('do not know how to compute derivative of %s' % str(self))

    def find_vf(self):
        return self.var.vform

    base_complexity = 0

def broadcast_expr(expr, shape):
    if len(shape) == 1:
        return LiteralVectorExpr(shape[0] * [expr])
    elif len(shape) == 2:
        return LiteralMatrixExpr(shape[0] * [shape[1] * [expr]])
    else:
        raise ValueError('invalid shape %s in broadcast_expr' % shape)

class NegExpr(Expr):
    def __init__(self, expr):
        if not expr.is_scalar(): raise TypeError('can only negate scalars')
        self.shape = ()
        self.children = (expr,)
    def __str__(self):
        return '-%s' % str(self.x)
    base_complexity = 0 # don't bother extracting subexpressions which are simple negation

class BuiltinFuncExpr(Expr):
    def __init__(self, funcname, expr):
        expr = as_expr(expr)
        if not expr.is_scalar(): raise TypeError('can only compute %s of scalars' % funcname)
        self.funcname = funcname
        self.shape = ()
        self.children = (expr,)
    def __str__(self):
        return '%s(%s)' % (self.funcname, str(self.x))

def OperExpr(oper, x, y):
    # coerce arguments to Expr, in case they are number literals
    x = as_expr(x)
    y = as_expr(y)

    if x.is_scalar() and y.is_scalar():
        return ScalarOperExpr(oper, x, y)
    elif len(x.shape) == len(y.shape):      # vec.vec or mat.mat
        return TensorOperExpr(oper, x, y)
    elif x.is_scalar() and not y.is_scalar():
        return OperExpr(oper, broadcast_expr(x, y.shape), y)
    elif not x.is_scalar() and y.is_scalar():
        return OperExpr(oper, x, broadcast_expr(y, x.shape))
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

    def _dx_impl(self, k, times, para):
        if self.oper == '+':
            return Dx(self.x, k, times, para) + Dx(self.y, k, times, para)
        elif self.oper == '-':
            return Dx(self.x, k, times, para) - Dx(self.y, k, times, para)
        elif self.oper == '*':
            assert times==1, 'higher-order derivative rules not implemented'
            return Dx(self.x, k, times, para) * self.y + self.x * Dx(self.y, k, times, para)
        elif self.oper == '/':
            assert times==1, 'higher-order derivative rules not implemented'
            return (Dx(self.x, k, times, para) * self.y -
                    self.x * Dx(self.y, k, times, para)) / (self.y * self.y)
        else:
            raise ValueError('do not know how to compute derivative for operation %s' % self.oper)

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
        if len(I) == 1:
            # for vector indexing, we use only the index, not a 1-tuple
            I = I[0]
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
        # NB: the `physical` argument is only meaningful if sum(D) > 0
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
        return (self.basisfun.hash(), self.D, self.physical)

    def depends(self):
        return set((self.basisfun,))

    def scope(self):
        return Scope.BASISFUN

    def without_derivs(self):
        """Return the underlying basis function without derivatives."""
        return PartialDerivExpr(self.basisfun, len(self.D) * (0,), physical=False)

    def is_physical_deriv(self):
        return self.physical

    def make_parametric(self):
        if self.physical:
            return PartialDerivExpr(self.basisfun, self.D, physical=False)
        else:
            return self

    def _dx_impl(self, k, times, parametric):
        Dnew = list(self.D)
        old_order = sum(Dnew)
        if bool(parametric) != (not self.physical) and old_order != 0:
            raise RuntimeError('cannot mix physical and parametric derivatives')
        Dnew[k] += times
        return PartialDerivExpr(self.basisfun, Dnew, physical=not parametric)

    def find_vf(self):
        return self.basisfun.vform


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

class GaussWeightExpr(Expr):
    def __init__(self, axis):
        self.axis = axis
        self.shape = ()
        self.children = ()
    def hash_key(self):
        return (self.axis,)
    def scope(self):
        return Scope.FIELD
    def __str__(self):
        return 'gw%d' % self.axis

class VolumeMeasureExpr(Expr):
    def __init__(self):
        self.shape = ()
        self.children = ()
    def scope(self):
        return Scope.FIELD
    def __str__(self):
        return 'dx'

class SurfaceMeasureExpr(Expr):
    def __init__(self):
        self.shape = ()
        self.children = ()
    def scope(self):
        return Scope.FIELD
    def __str__(self):
        return 'ds'

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
                result.append(fun(e))
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
    if hasattr(expr, 'oper'):
        s = '(%s)' % expr.oper
    elif expr.is_var_expr() or isinstance(expr, VolumeMeasureExpr) or isinstance(expr, PartialDerivExpr) or isinstance(expr, ConstExpr):
        s = str(expr)
    elif isinstance(expr, BuiltinFuncExpr):
        s = expr.funcname
    else:
        s = type(expr).__name__

    if data is None:
        print(indent + s)
    else:
        print(indent + s + ' (%s)' % data[expr])

    for c in expr.children:
        tree_print(c, data, indent + '  ')

def exprhash(expr):
    return expr.hash(tuple(exprhash(c) for c in expr.children))

def _to_literal_vec_mat(e):
    """Convert all matrix and vector expressions into literal expressions,
    i.e., into elementwise scalar expressions."""
    if e.is_var_expr():
        return
    if e.is_vector() and not isinstance(e, LiteralVectorExpr):
        return LiteralVectorExpr(e)
    if e.is_matrix() and not isinstance(e, LiteralMatrixExpr):
        return LiteralMatrixExpr(
                [[e[i,j] for j in range(e.shape[1])]
                    for i in range(e.shape[0])])

# expression manipulation functions ############################################
# notation is as close to UFL as possible

#: A symbolic expression representing the integration weight stemming from the
#: geometry map.
dx = VolumeMeasureExpr()

#: A symbolic expression representing the integration weight stemming from the
#: geometry map for surface integrals.
ds = SurfaceMeasureExpr()

def Dx(expr, k, times=1, parametric=False):
    """Partial derivative of `expr` along the `k`-th coordinate axis."""
    expr = as_expr(expr)
    if hasattr(expr, '_dx_impl'):
        return expr._dx_impl(k, times, parametric)
    elif expr.is_vector():
        return LiteralVectorExpr(Dx(z, k, times, parametric=parametric)
                for z in expr)
    elif expr.is_matrix():
        raise NotImplementedError('derivative of matrix not implemented')
    else:   # scalar
        raise TypeError('do not know how to compute derivative of %s ' % type(expr))

def Dt(expr, times=1):
    expr = as_expr(expr)
    if expr.is_vector():
        return LiteralVectorExpr(Dt(z, times) for z in expr)
    elif expr.is_matrix():
        raise NotImplementedError('time derivative of matrix not implemented')
    else:   # scalar
        vf = expr.find_vf()
        if not vf:
            raise ValueError('could not determine ambient VForm')
        if not vf.spacetime:
            raise TypeError('can only compute time derivatives in spacetime assemblers')
        return Dx(expr, vf.timedim, times)

def grad(expr, dims=None, parametric=False):
    """Gradient of an expression.

    If `expr` is scalar, results in a vector of all partial derivatives.

    If `expr` is a vector, results in the Jacobian matrix whose rows are
    the gradients of the individual components.

    If `dims` is specified, it is a tuple of dimensions along which to take
    the derivative. By default, all space dimensions are used.

    If `parametric` is true, the gradient with respect to the coordinates in
    the parameter domain is computed. By default, the gradient is computed in
    physical coordinates (transformed by the geometry map).
    """
    expr = as_expr(expr)
    if expr.is_scalar():
        if dims is None:
            vf = expr.find_vf()
            if not vf:
                raise ValueError('could not automatically determine dimensions - please specify dims')
            dims = vf.spacedims
        return as_vector(Dx(expr, k, parametric=parametric) for k in dims)
    elif expr.is_vector():
        # compute Jacobian of vector expression
        return as_matrix([grad(z, dims=dims, parametric=parametric) for z in expr])
    else:
        raise TypeError('cannot compute gradient for expr of shape %s' % expr.shape)

def hess(expr, parametric=False):
    """Hessian matrix of a scalar expression.

    If `parametric` is true, the Hessian with respect to the coordinates in
    the parameter domain is computed. By default, the Hessian is computed in
    physical coordinates (transformed by the geometry map).
    """
    expr = as_expr(expr)
    if expr.is_scalar():
        return grad(grad(expr, parametric=parametric), parametric=parametric)
    else:
        raise TypeError('cannot compute Hessian for expr of shape %s' % expr.shape)

def div(expr, parametric=False):
    """The divergence of a vector-valued expressions, resulting in a scalar."""
    expr = as_expr(expr)
    if not expr.is_vector():
        raise TypeError('can only compute divergence of vector expression')
    return tr(grad(expr, parametric=parametric))

def curl(expr):
    """The curl (or rot) of a 3D vector expression."""
    expr = as_expr(expr)
    if not (expr.is_vector() and len(expr) == 3):
        raise TypeError('can only compute curl of 3D vector expression')
    return as_vector((
        expr[2].dx(1) - expr[1].dx(2),
        expr[0].dx(2) - expr[2].dx(0),
        expr[1].dx(0) - expr[0].dx(1),
    ))

def as_expr(x):
    """Interpret input as an expression; useful for constants."""
    if isinstance(x, Expr):
        return x
    elif isinstance(x, numbers.Number):
        return ConstExpr(x)
    elif isinstance(x, tuple):
        if all(isinstance(z, numbers.Number) or isinstance(z, Expr) for z in x):
            return as_vector(x)
    raise TypeError('cannot coerce {} to expression'.format(x))

def as_vector(x):
    """Convert a sequence of expressions to a vector expression."""
    return LiteralVectorExpr(x)
def as_matrix(x):
    """Convert a sequence of sequence of expressions to a matrix expression."""
    return LiteralMatrixExpr(x)

def inner(x, y):
    """The inner product of two vector or matrix expressions."""
    x = as_expr(x)
    y = as_expr(y)
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
    a = as_expr(a)
    b = as_expr(b)
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
    if n == 1:
        return as_matrix([[invdet]])
    cofacs = as_matrix(
            [[ (-1)**(i+j) * minor(A, i, j)
                for i in range(n)]
                for j in range(n)])
    return invdet * cofacs

def cross(x, y):
    """Cross product of two 3D vectors."""
    x = as_expr(x)
    y = as_expr(y)
    return VectorCrossExpr(x, y)

def outer(x, y):
    """Outer product of two vectors, resulting in a matrix."""
    x = as_expr(x)
    y = as_expr(y)
    return OuterProdExpr(x, y)

def norm(x):
    """Euclidean norm of a vector."""
    x = as_expr(x)
    if not x.is_vector():
        raise TypeError('expression is not a vector')
    return sqrt(inner(x, x))

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

def mass_vf(dim, c=1):
    V = VForm(dim)
    u, v = V.basisfuns()
    V.add(c * u * v * dx)
    return V

def stiffness_vf(dim, a=1):
    V = VForm(dim)
    u, v = V.basisfuns()
    B = V.let('B', V.W * dot(V.JacInv, V.JacInv.T), symmetric=True)
    V.add(a*B.dot(grad(u, parametric=True)).dot(grad(v, parametric=True)))
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

def L2functional_vf(dim, physical=False, updatable=False):
    V = VForm(dim, arity=1)
    v = V.basisfuns()
    f = V.input('f', shape=(), physical=physical, updatable=updatable)
    V.add(f * v * dx)
    return V

#surface variational forms

def mass_Bvf(dim):
    V = VForm(dim, boundary=True)
    u, v = V.basisfuns()
    V.add(u * v * ds)
    return V

def L2functional_Bvf(dim, physical=False, updatable=False):
    V = VForm(dim, arity=1, boundary=True)
    v = V.basisfuns()
    f = V.input('f', shape=(), physical=physical, updatable=updatable)
    V.add(f * v * ds)
    return V
################################################################################
# parse strings to VForms
################################################################################

def _check_input_field(kvs, geo, f):
    # return (shape, physical) pair for function f
    # NB: by default, _BaseGeoFuncs are considered parametric and explicitly
    #     given functions physical!
    from . import bspline
    if isinstance(f, bspline._BaseGeoFunc):
        return f.output_shape(), False
    else:   # assume a callable function
        supp = tuple(kv.support() for kv in kvs)
        mid = tuple((a+b)/2 for (a,b) in supp)
        result = f(*geo(*mid))        # evaluate it at the midpoint
        return np.shape(result), True

def parse_vf(expr, kvs, args=dict(), bfuns=None, boundary=False, updatable=[]):
    from . import bspline
    def is_tp_spl(x):
        return all(isinstance(y, bspline.KnotVector) for y in x)
    if is_tp_spl(kvs):
        pass        # a single TP spline space was passed
    elif is_tp_spl(kvs[0]):
        kvs = kvs[0]    # multiple spaces are being used; we only need one of them
    else:
        raise ValueError('expected a tensor product spline space in `kvs`')

    dim = len(kvs)
    loc = dict()
    geo=args['geo']

    # parse all identifiers in the expression string
    import re
    words = set(re.findall(r"[^\d\W]\w*", expr))

    if bfuns is None:
        # check which of 'u' and 'v' was used
        candidate_bfuns = set(('u', 'v'))
        bfuns = [(bf, 1, 0) for bf in sorted(words & candidate_bfuns)]
    else:
        # normalize the bfun representation
        bfuns_new = []
        for bf in bfuns:
            if isinstance(bf, str):
                bf = (bf, 1, 0)
            if len(bf) == 1:
                bf = bf + (1,)      # make it a scalar basis function
            if len(bf) == 2:
                bf = bf + (0,)      # by default, lives in space 0
            bfuns_new.append(bf)
        bfuns = bfuns_new

    # determine volume/surface integral
    geo_dim = dim
    if 'ds' in words:
        if 'dx' in words:
            raise RuntimeError("got both 'dx' and 'ds' - is this a volume or a surface integral?")
        if not boundary:
            geo_dim += 1

    arity = len(bfuns)
    if not arity in (1, 2):
        raise ValueError('arity should be 1 or 2')
    vf = VForm(dim=dim, geo_dim=geo_dim, boundary=boundary, arity=arity)

    # set up basis functions
    components = tuple(bf[1] for bf in bfuns)
    if all(c == 1 for c in components):
        components = len(components) * (None,)  # force scalar assembler
    spaces = tuple(bf[2] for bf in bfuns)

    if arity == 1:
        v = vf.basisfuns(components=components, spaces=spaces)
        loc[bfuns[0][0]] = v
    elif arity == 2:
        u, v = vf.basisfuns(components=components, spaces=spaces)
        loc[bfuns[0][0]] = u
        loc[bfuns[1][0]] = v

    # set up used input functions and constant parameters
    for inp in sorted(set(args.keys()) & words):
        upd = (inp in updatable)
        if callable(args[inp]):     # function - treat as input field
            shp, phys = _check_input_field(kvs, geo, args[inp])
            loc[inp] = vf.input(inp, shape=shp, physical=phys, updatable=upd)
        else:       # constant value or array - treat as parameter
            shp = np.shape(args[inp])
            loc[inp] = vf.parameter(inp, shape=shp)

    # set up additional terms
    if 'x' in words and 'x' not in args:
        loc['x'] = vf.Geo       # x is a shorthand for the physical coordinates
    if 'n' in words and 'n' not in args:
        loc['n'] = vf.normal
    if 'gw' in words and 'gw' not in args:
        loc['gw'] = vf.GaussWeight
    if 'jac' in words and 'jac' not in args:
        loc['jac'] = vf.Jac

    vf.add(eval(expr, globals(), loc))
    return vf
