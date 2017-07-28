import os.path
from jinja2 import Template
from collections import OrderedDict
from functools import reduce
import operator
import numpy as np
import networkx

class PyCode:
    def __init__(self):
        self._lines = []
        self._indent = ''

    def indent(self, num=1):
        self._indent += num * '    '

    def dedent(self, num=1):
        self._indent = self._indent[(4*num):]

    def put(self, s):
        if len(s) == 0:
            self._lines.append('')
        else:
            self._lines.append(self._indent + s)

    def putf(self, s, **kwargs):
        self.put(s.format(**kwargs))

    def declare_local_variable(self, type, name, init=None):
        if init is not None:
            self.putf('cdef {type} {name} = {init}', type=type, name=name, init=init)
        else:
            self.putf('cdef {type} {name}', type=type, name=name)

    def for_loop(self, idx, upper):
        self.putf('for {idx} in range({upper}):', idx=idx, upper=upper)
        self.indent()

    def end_loop(self):
        self.dedent()

    def result(self):
        return '\n'.join(self._lines)


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
    def __init__(self, name, vform):
        self.name = name
        self.vform = vform
        self.asmgen = None  # to be set to AsmGenerator for code generation

class VForm:
    """Abstract representation of a variational form in matrix-vector form."""
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
        self.basis_funs = (BasisFun('u', self), BasisFun('v', self))
        # variables provided during initialization (geo_XXX)
        self.init_det = False
        self.init_jac = False
        self.init_jacinv = False
        self.init_weights = False       # geo_weights = Gauss weights * abs(geo_det)
        # predefined local variables with their generators (created on demand)
        self.predefined_vars = {
            'u':     lambda self: self.basisval(self.basis_funs[0]),
            'v':     lambda self: self.basisval(self.basis_funs[1]),
            'gu':    lambda self: grad(self.u),
            'gv':    lambda self: grad(self.v),
            'gradu': lambda self: matmul(self.JacInv.T[self.spacedims, self.spacedims], self.gu),
            'gradv': lambda self: matmul(self.JacInv.T[self.spacedims, self.spacedims], self.gv),
        }
        self.cached_vars = {}

    def field_vars(self):
        return (var for var in self.vars.values() if var.is_field())

    def register_scalar_field(self, name, src=''):
        self.vars[name] = AsmVar(name, src=src, shape=(), local=False)

    def register_vector_field(self, name, size=None, src=''):
        if size is None: size = self.dim
        self.vars[name] = AsmVar(name, src=src, shape=(size,), local=False)

    def register_matrix_field(self, name, shape=None, symmetric=False, src=''):
        if shape is None: shape = (self.dim, self.dim)
        assert len(shape) == 2
        self.vars[name] = AsmVar(name, src=src, shape=shape, local=False, symmetric=symmetric)
        return NamedMatrixExpr(name, shape=shape, symmetric=symmetric)

    def declare_sourced_var(self, name, shape, src, symmetric=False):
        var = AsmVar(name, src=src, shape=shape, local=True, symmetric=symmetric)
        self.vars[name] = var
        if shape is ():
            return NamedScalarExpr(var)
        elif len(shape) == 1:
            return NamedVectorExpr(var, shape)
        elif len(shape) == 2:
            return NamedMatrixExpr(var, shape, symmetric=symmetric)

    def add(self, expr):
        assert not self.vec, 'add() is only for scalar assemblers; use add_at()'
        assert expr.is_scalar(), 'require scalar expression'
        self.exprs.append(expr)

    def add_at(self, idx, expr):
        assert self.vec, 'add_at() is only for vector assemblers; use add()'
        assert expr.is_scalar(), 'require scalar expression'
        self.exprs.append((idx, expr))

    def partial_deriv(self, basisfun, D):
        """Parametric partial derivative of `basisfun` of order `D=(Dx1, ..., Dxd)`"""
        return PartialDerivExpr(basisfun, D)

    def basisval(self, basisfun):
        return self.partial_deriv(basisfun, self.dim * (0,))

    def gradient(self, basisfun, dims=None, additional_derivs=None):
        if dims is None:
            dims = range(self.dim)
        if additional_derivs is None:
            additional_derivs = self.dim * [0]

        entries = []
        for k in dims:
            D = list(additional_derivs)
            D[k] += 1
            entries.append(self.partial_deriv(basisfun, D))
        return LiteralVectorExpr(entries)

    def let(self, varname, expr, symmetric=False):
        var = AsmVar(varname, expr, shape=None, local=True, symmetric=symmetric)
        self.vars[varname] = var
        return named_expr(var, shape=expr.shape, symmetric=symmetric)

    # automatically produce caching getters for predefined on-demand local variables
    def __getattr__(self, name):
        if name in self.predefined_vars:
            if not name in self.cached_vars:
                self.cached_vars[name] = self.let(name, self.predefined_vars[name](self))
            return self.cached_vars[name]
        else:
            msg = "'{0}' object has no attribute '{1}'"
            raise AttributeError(msg.format(type(self).__name__, name))

    # on-demand field variables (cannot currently be autogenerated)

    @property
    def W(self):
        if not 'W' in self.cached_vars:
            self.init_weights = True
            self.cached_vars['W'] = self.declare_sourced_var('W', shape=(), src='geo_weights')
        return self.cached_vars['W']

    @property
    def JacInv(self):
        if not 'JacInv' in self.cached_vars:
            self.init_jacinv = True
            self.cached_vars['JacInv'] = self.declare_sourced_var('JacInv', shape=(self.dim,self.dim), src='geo_jacinv')
        return self.cached_vars['JacInv']

    @property
    def JacInv_x(self):
        "Inverse Jacobian only for the space dimensions. Assumes space-time cylinder."""
        if not self.spacetime:
            raise TypeError('JacInv_x only defined for spacetime assemblers')
        return self.JacInv[self.spacedims, self.spacedims]

    def dependency_graph(self):
        G = networkx.DiGraph()
        for var in self.vars.values():
            G.add_node(var.name)
            if var.expr:
                for dep in var.expr.depends():
                    G.add_edge(dep, var.name)

        # compute transitive dependencies of all used expressions
        expr_deps = set()
        for expr in self.root_exprs():
            for dep in expr.depends():
                expr_deps.add(dep)
                expr_deps |= networkx.ancestors(G, dep)

        # keep only those vars which are actually used (sorted for code stability)
        return G.subgraph(sorted(expr_deps))

    def as_vars(self, vars):
        def to_var(v):
            return v if isinstance(v, AsmVar) else self.vars[v]
        return [to_var(v) for v in vars]

    def transitive_deps(self, vars):
        """Return all vars on which the given vars depend directly or indirectly."""
        vars = self.as_vars(vars)
        deps = reduce(operator.or_,
                (networkx.ancestors(self.dep_graph, var.name) for var in vars),
                set())
        deps -= {'@u', '@v'}    # remove virtual nodes
        # return in sorted order to increase code stability
        return [self.vars[name] for name in sorted(deps)]

    def linearize_vars(self, vars):
        """Returns an admissible order for computing the given vars, i.e., which
        respects the dependency relation."""
        names = [v.name for v in self.as_vars(vars)]
        return [self.vars[vn] for vn in self.linear_deps if vn in names]

    def vars_without_dep_on(self, exclude):
        """Return a linearized list of all expr vars which do not depend on the given vars."""
        G = self.dep_graph
        nodes = set(G.nodes())
        # remove 'exlude' vars and anything that depends on them
        for var in exclude:
            nodes.discard(var)
            nodes -= networkx.descendants(G, var)
        # whatever remains and has an expr is a pure field variable and can be precomputed
        precomp_vars = {var for var in nodes if self.vars[var].expr}
        return self.linearize_vars(precomp_vars)

    def root_exprs(self):
        if self.vec:
            return (expr for idx,expr in self.exprs)
        else:
            return self.exprs

    def dependency_analysis(self):
        self.dep_graph = self.dependency_graph()
        self.linear_deps = networkx.topological_sort(self.dep_graph)

        # determine precomputable vars (no dependency on basis functions)
        self.precomp = self.vars_without_dep_on(('@u', '@v'))
        self.precomp_deps = self.transitive_deps(self.precomp)
        for var in self.precomp:
            # remove all dependencies since it's now precomputed
            # this ensures kernel_deps will not depend on dependencies of precomputed vars
            self.dep_graph.remove_edges_from(self.dep_graph.in_edges([var.name]))

        # compute linearized list of vars the kernel depends on
        kernel_deps = reduce(operator.or_,
                (set(expr.depends()) for expr in self.root_exprs()), set())
        kernel_deps = self.as_vars(kernel_deps - {'@u', '@v'})
        kernel_deps = self.transitive_deps(kernel_deps) + kernel_deps
        self.kernel_deps = self.linearize_vars(kernel_deps)

        # promote precomputed/manually sourced dependencies to field variables
        for var in self.kernel_deps:
            if var.src or var in self.precomp:
                var.local = False

    def iterexpr(self, expr, deep=False, type=None):
        """Iterate through all subexpressions of `expr` in depth-first order.

        If `deep=True`, follow named variable references.
        If `type` is given, only exprs which are instances of that type are yielded.
        """
        for c in expr.children:
            yield from self.iterexpr(c, deep=deep, type=type)
        if (deep
                and expr.is_named_expr()
                and expr.var.expr is not None):
            yield from self.iterexpr(expr.var.expr, deep=deep, type=type)
        else:
            if type is None or isinstance(expr, type):
                yield expr

    def all_exprs(self, type=None):
        """Deep, depth-first iteration of all expressions with dependencies.

        If `type` is given, only exprs which are instances of that type are yielded.
        """
        for expr in self.root_exprs():
            yield from self.iterexpr(expr, deep=True, type=type)

    def find_max_deriv(self):
        return max(max(e.D) for e in self.all_exprs(type=PartialDerivExpr))



# Code generation #############################################################

class AsmGenerator:
    """Generates a Cython assembler class from an abstract :class:`VForm`."""
    def __init__(self, vform, classname, code):
        self.vform = vform
        self.classname = classname
        self.code = code
        self.dim = self.vform.dim
        self.vec = self.vform.vec

        # fixup PartialDerivExprs for code generation
        for bf in self.vform.basis_funs:
            bf.asmgen = self

    def indent(self, num=1):
        self.code.indent(num)

    def dedent(self, num=1):
        self.code.dedent(num)

    def end_function(self):
        self.dedent()

    def put(self, s):
        self.code.put(s)

    def putf(self, s, **kwargs):
        env = dict(self.env)
        env.update(kwargs)
        self.code.putf(s, **env)

    def dimrep(self, s, sep=', '):
        return sep.join([s.format(k, **self.env) for k in range(self.dim)])

    def extend_dim(self, i):
        # ex.: dim = 3, i = 1  ->  'None,:,None'
        slices = self.dim * ['None']
        slices[i] = ':'
        return ','.join(slices)

    def tensorprod(self, var):
        return ' * '.join(['{0}[{1}][{2}]'.format(var, k, self.extend_dim(k))
            for k in range(self.dim)])

    def gen_pderiv(self, basisfun, D, idx='i'):
        """Generate code for computing parametric partial derivative of `basisfun` of order `D=(Dx1, ..., Dxd)`"""
        D = tuple(reversed(D))  # x is last axis
        assert len(D) == self.dim
        assert all(0 <= d <= self.numderiv for d in D)
        factors = [
                "VD{var}{k}[{nderiv}*{idx}{k}+{ofs}]".format(
                    var = basisfun.name,
                    idx = idx,
                    k   = k,
                    ofs = D[k],
                    nderiv = self.numderiv+1, # includes 0-th derivative
                )
                for k in range(self.dim)]
        return '(' + ' * '.join(factors) + ')'

    def declare_index(self, name, init=None):
        self.code.declare_local_variable('size_t', name, init)

    def declare_scalar(self, name, init=None):
        self.code.declare_local_variable('double', name, init)

    def declare_pointer(self, name, init=None):
        self.code.declare_local_variable('double*', name, init)

    def declare_vec(self, name, size=None):
        if size is None:
            size = self.dim
        self.putf('cdef double {name}[{size}]', name=name, size=size)

    def gen_assign(self, var, expr):
        if expr.is_vector():
            for k in range(expr.shape[0]):
                self.putf('{name}[{k}] = {rhs}',
                        name=var.name,
                        k=k,
                        rhs=expr[k].gencode())
        elif expr.is_matrix():
            m, n = expr.shape
            for i in range(m):
                for j in range(n):
                    if var.symmetric and i > j:
                        continue
                    self.putf('{name}[{k}] = {rhs}',
                            name=var.name,
                            k=i*m + j,
                            rhs=expr[i,j].gencode())
        else:
            self.put(var.name + ' = ' + expr.gencode())

    def cython_pragmas(self):
        self.put('@cython.boundscheck(False)')
        self.put('@cython.wraparound(False)')
        self.put('@cython.initializedcheck(False)')

    def field_type(self, var):
        return 'double[{X}:1]'.format(X=', '.join((self.dim + len(var.shape)) * ':'))

    def declare_var(self, var, ref=False):
        if ref:
            if var.is_scalar():
                self.declare_scalar(var.name)
            elif var.is_vector() or var.is_matrix():
                self.declare_pointer(var.name)
        else:   # no ref - declare local storage
            if var.is_vector():
                self.declare_vec(var.name, size=var.shape[0])
            elif var.is_matrix():
                self.declare_vec(var.name, size=var.shape[0]*var.shape[1])
            else:
                self.declare_scalar(var.name)

    def load_field_var(self, var, I, ref_only=False):
        if var.is_scalar():
            if not ref_only: self.putf('{name} = _{name}[{I}]', name=var.name, I=I)
        elif var.is_vector():
            self.putf('{name} = &_{name}[{I}, 0]', name=var.name, I=I)
        elif var.is_matrix():
            self.putf('{name} = &_{name}[{I}, 0, 0]', name=var.name, I=I)

    def start_loop_with_fields(self, fields_in, fields_out=[]):
        fields = fields_in + fields_out

        # get input size from an arbitrary field variable
        for k in range(self.dim):
            self.declare_index(
                    'n%d' % k,
                    '_{var}.shape[{k}]'.format(k=k, var=fields[0].name)
            )
        self.put('')

        # temp storage for field variables
        for var in fields:
            self.declare_var(var, ref=True)

        # declare iteration indices
        for k in range(self.dim):
            self.declare_index('i%d' % k)

        for k in range(self.dim):
            self.code.for_loop('i%d' % k, 'n%d' % k)

        # generate assignments for field variables
        I = self.dimrep('i{}')  # current grid index
        for var in fields_in:
            self.load_field_var(var, I)
        for var in fields_out:
            # these have no values yet, only get a reference
            self.load_field_var(var, I, ref_only=True)
        self.put('')

    def generate_kernel(self):
        # function definition
        self.cython_pragmas()
        self.put('@staticmethod')
        rettype = 'void' if self.vec else 'double'
        self.putf('cdef {rettype} combine(', rettype=rettype)
        self.indent(2)

        field_params = [var for var in self.vform.kernel_deps if var.is_field()]
        local_vars   = [var for var in self.vform.kernel_deps if var.is_local()]

        # parameters
        for var in field_params:
            self.putf('{type} _{name},', type=self.field_type(var), name=var.name)

        self.put(self.dimrep('double* VDu{}') + ',')
        self.put(self.dimrep('double* VDv{}') + ',')
        if self.vec:    # for vector assemblers, result is passed as a pointer
            self.put('double result[]')
        self.dedent()
        self.put(') nogil:')

        # local variables
        if not self.vec:    # for vector assemblers, result is passed as a pointer
            self.declare_scalar('result', '0.0')

        # temp storage for local variables
        for var in local_vars:
            self.declare_var(var)

        self.declare_custom_variables()

        self.put('')

        ############################################################
        # main loop over all Gauss points
        self.start_loop_with_fields(field_params)

        # generate code for computing local variables
        for var in local_vars:
            self.gen_assign(var, var.expr)

        # if needed, generate custom code for the bilinear form a(u,v)
        self.generate_biform_custom()

        # generate code for all expressions in the bilinear form
        if self.vec:
            for idx, expr in self.vform.exprs:
                self.put(('result[%d] += ' % idx) + expr.gencode())
        else:
            for expr in self.vform.exprs:
                self.put('result += ' + expr.gencode())

        # end main loop
        for _ in range(self.dim):
            self.code.end_loop()
        ############################################################

        if not self.vec:
            self.put('return result')
        self.end_function()

    def gen_assemble_impl_header(self):
        if self.vec:
            funcdecl = 'cdef void assemble_impl(self, size_t[{dim}] i, size_t[{dim}] j, double result[]) nogil:'.format(dim=self.dim)
        else:
            funcdecl = 'cdef double assemble_impl(self, size_t[{dim}] i, size_t[{dim}] j) nogil:'.format(dim=self.dim)
        zeroret = '' if self.vec else '0.0'  # for vector assemblers, result[] is 0-initialized

        self.cython_pragmas()
        src = r"""{funcdecl}
    cdef int k
    cdef IntInterval intv
    cdef size_t g_sta[{dim}]
    cdef size_t g_end[{dim}]
    cdef (double*) values_i[{dim}]
    cdef (double*) values_j[{dim}]

    for k in range({dim}):
        intv = intersect_intervals(make_intv(self.meshsupp[k][i[k],0], self.meshsupp[k][i[k],1]),
                                   make_intv(self.meshsupp[k][j[k],0], self.meshsupp[k][j[k],1]))
        if intv.a >= intv.b:
            return {zeroret}      # no intersection of support
        g_sta[k] = self.nqp * intv.a    # start of Gauss nodes
        g_end[k] = self.nqp * intv.b    # end of Gauss nodes

        values_i[k] = &self.C[k][ i[k], g_sta[k], 0 ]
        values_j[k] = &self.C[k][ j[k], g_sta[k], 0 ]

""".format(
            dim=self.dim,
            funcdecl=funcdecl,
            zeroret=zeroret
        )
        for line in src.splitlines():
            self.put(line)
        self.indent()

    def generate_assemble_impl(self):
        self.gen_assemble_impl_header()

        # generate call to assembler kernel
        if self.vec:
            self.putf('{classname}{dim}D.combine(', classname=self.classname)
        else:
            self.putf('return {classname}{dim}D.combine(', classname=self.classname)
        self.indent(2)

        # generate field variable arguments
        idx = self.dimrep('g_sta[{0}]:g_end[{0}]')
        for var in self.vform.kernel_deps:
            if var.is_field():
                self.putf('self.{name} [ {idx} ],', name=var.name, idx=idx)

        # generate basis function value arguments
        # a_ij = a(phi_j, phi_i)  -- pass values for j (trial function) first
        self.put(self.dimrep('values_j[{0}]') + ',')
        self.put(self.dimrep('values_i[{0}]') + ',')

        # generate output argument if needed (for vector assemblers)
        if self.vec:
            self.put('result')

        self.dedent(2)
        self.put(')')
        self.end_function()

    def generate_init(self):
        vf = self.vform

        self.put('def __init__(self, kvs, geo):')
        self.indent()
        for line in \
"""assert geo.dim == {dim}, "Geometry has wrong dimension"
self.base_init(kvs)

gaussgrid, gaussweights = make_tensor_quadrature([kv.mesh for kv in kvs], self.nqp)
N = tuple(gg.shape[0] for gg in gaussgrid)  # grid dimensions

self.C = compute_values_derivs(kvs, gaussgrid, derivs={maxderiv})""".splitlines():
            self.putf(line)
        self.put('')

        # dependencies
        if vf.init_weights: vf.init_det = True
        if vf.init_jacinv or vf.init_det: vf.init_jac = True

        # compute Jacobian
        if vf.init_jac:
            self.put('geo_jac = geo.grid_jacobian(gaussgrid)')

        # determinant and/or inverse of Jacobian
        if vf.init_det and vf.init_jacinv:
            self.put('geo_det, geo_jacinv = det_and_inv(geo_jac)')
        elif vf.init_det:
            self.put('geo_det = determinants(geo_jac)')
        elif vf.init_jacinv:
            self.put('geo_jacinv = inverses(geo_jac)')

        if vf.init_weights:
            self.put('geo_weights = %s * np.abs(geo_det)' % self.tensorprod('gaussweights'))

        for var in vf.field_vars():
            if var.src:
                self.putf("self.{name} = {src}", name=var.name, src=var.src)
            elif var.expr:  # custom precomputed field var
                self.putf("self.{name} = np.empty(N + {shape})", name=var.name, shape=var.shape)

        if vf.precomp:
            # call precompute function
            self.putf('{classname}{dim}D.precompute_fields(', classname=self.classname)
            self.indent(2)
            for var in vf.precomp_deps + vf.precomp:
                if var.src:
                    self.put(var.src + ',')
                else:
                    self.put('self.' + var.name + ',')
            self.dedent(2)
            self.put(')')

        self.initialize_custom_fields()
        self.end_function()

    def generate_precomp(self):
        vf = self.vform

        self.cython_pragmas()
        self.put('@staticmethod')
        self.put('cdef void precompute_fields(')
        self.indent(2)
        self.put('# input')
        for var in vf.precomp_deps:
            self.putf('{type} _{name},', type=self.field_type(var), name=var.name)
        self.put('# output')
        for var in vf.precomp:
            self.putf('{type} _{name},', type=self.field_type(var), name=var.name)
        self.dedent()
        self.put(') nogil:')

        self.start_loop_with_fields(vf.precomp_deps, vf.precomp)

        for var in vf.precomp:
            self.gen_assign(var, var.expr)
            if var.is_scalar():
                # for scalars, we need to explicitly copy the computed value into
                # the field array; vectors and matrices use pointers directly
                self.putf('_{name}[{I}] = {name}', name=var.name, I=I)

        # end main loop
        for _ in range(self.dim):
            self.code.end_loop()
        self.end_function()

    # main code generation entry point

    def generate(self):
        self.vform.dependency_analysis()
        self.numderiv = self.vform.find_max_deriv()

        self.env = {
            'dim': self.vform.dim,
            'maxderiv': self.numderiv,
        }

        baseclass = 'BaseVectorAssembler' if self.vec else 'BaseAssembler'
        self.putf('cdef class {classname}{dim}D({base}{dim}D):',
                classname=self.classname, base=baseclass)
        self.indent()

        # declare instance variables
        self.put('cdef vector[double[:, :, ::1]] C       # 1D basis values. Indices: basis function, mesh point, derivative')
        # declare field variables
        for var in self.vform.field_vars():
            self.putf('cdef double[{X}:1] {name}',
                    X=', '.join((self.dim + len(var.shape)) * ':'),
                    name=var.name)
        self.put('')

        # generate methods
        self.generate_init()
        self.put('')
        if self.vform.precomp:
            self.generate_precomp()
            self.put('')
        self.generate_kernel()
        self.put('')
        self.generate_assemble_impl()
        self.dedent()
        self.put('')

    # hooks for custom code generation

    def declare_custom_variables(self):
        pass

    def initialize_custom_fields(self):
        pass

    def generate_biform_custom(self):
        pass



tmpl_generic = Template(r'''
################################################################################
# {{DIM}}D Assemblers
################################################################################

cdef class BaseAssembler{{DIM}}D:
    cdef int nqp
    cdef size_t[{{DIM}}] ndofs
    cdef int[{{DIM}}] p
    cdef vector[ssize_t[:,::1]] meshsupp
    cdef list _asm_pool     # list of shared clones for multithreading

    cdef void base_init(self, kvs):
        assert len(kvs) == {{DIM}}, "Assembler requires two knot vectors"
        self.nqp = max([kv.p for kv in kvs]) + 1
        self.ndofs[:] = [kv.numdofs for kv in kvs]
        self.p[:]     = [kv.p for kv in kvs]
        self.meshsupp = [kvs[k].mesh_support_idx_all() for k in range({{DIM}})]
        self._asm_pool = []

    cdef _share_base(self, BaseAssembler{{DIM}}D asm):
        asm.nqp = self.nqp
        asm.ndofs[:] = self.ndofs[:]
        asm.meshsupp = self.meshsupp

    cdef BaseAssembler{{DIM}}D shared_clone(self):
        return self     # by default assume thread safety

    cdef inline size_t to_seq(self, size_t[{{DIM}}] ii) nogil:
        # by convention, the order of indices is (y,x)
        return {{ to_seq(indices, ndofs) }}

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef inline void from_seq(self, size_t i, size_t[{{DIM}}] out) nogil:
        {%- for k in range(1, DIM)|reverse %}
        out[{{k}}] = i % self.ndofs[{{k}}]
        i /= self.ndofs[{{k}}]
        {%- endfor %}
        out[0] = i

    cdef double assemble_impl(self, size_t[{{DIM}}] i, size_t[{{DIM}}] j) nogil:
        return -9999.99  # Not implemented

    cpdef double assemble(self, size_t i, size_t j):
        cdef size_t[{{DIM}}] I, J
        with nogil:
            self.from_seq(i, I)
            self.from_seq(j, J)
            return self.assemble_impl(I, J)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void multi_assemble_chunk(self, size_t[:,::1] idx_arr, double[::1] out) nogil:
        cdef size_t[{{DIM}}] I, J
        cdef size_t k

        for k in range(idx_arr.shape[0]):
            self.from_seq(idx_arr[k,0], I)
            self.from_seq(idx_arr[k,1], J)
            out[k] = self.assemble_impl(I, J)

    def multi_assemble(self, indices):
        """Assemble all entries given by `indices`.

        Args:
            indices: a sequence of `(i,j)` pairs or an `ndarray`
            of size `N x 2`.
        """
        cdef size_t[:,::1] idx_arr
        if isinstance(indices, np.ndarray):
            idx_arr = np.asarray(indices, order='C', dtype=np.uintp)
        else:   # possibly given as iterator
            idx_arr = np.array(list(indices), dtype=np.uintp)

        cdef double[::1] result = np.empty(idx_arr.shape[0])

        num_threads = pyiga.get_max_threads()
        if num_threads <= 1:
            self.multi_assemble_chunk(idx_arr, result)
        else:
            thread_pool = get_thread_pool()
            if not self._asm_pool:
                self._asm_pool = [self] + [self.shared_clone()
                        for i in range(1, thread_pool._max_workers)]

            results = thread_pool.map(_asm_chunk_{{DIM}}d,
                        self._asm_pool,
                        chunk_tasks(idx_arr, num_threads),
                        chunk_tasks(result, num_threads))
            list(results)   # wait for threads to finish
        return result

cpdef void _asm_chunk_{{DIM}}d(BaseAssembler{{DIM}}D asm, size_t[:,::1] idxchunk, double[::1] out):
    with nogil:
        asm.multi_assemble_chunk(idxchunk, out)


@cython.boundscheck(False)
@cython.wraparound(False)
cdef object generic_assemble_core_{{DIM}}d(BaseAssembler{{DIM}}D asm, bidx, bint symmetric=False):
    cdef unsigned[:, ::1] {{ dimrepeat('bidx{}') }}
    cdef long {{ dimrepeat('mu{}') }}, {{ dimrepeat('MU{}') }}
    cdef double[{{ dimrepeat(':') }}:1] entries

    {{ dimrepeat('bidx{}') }} = bidx
    {{ dimrepeat('MU{}') }} = {{ dimrepeat('bidx{}.shape[0]') }}

    cdef size_t[::1] {{ dimrepeat('transp{}') }}
    if symmetric:
    {%- for k in range(DIM) %}
        transp{{k}} = get_transpose_idx_for_bidx(bidx{{k}})
    {%- endfor %}
    else:
        {{ dimrepeat('transp{}', sep=' = ') }} = None

    entries = np.zeros(({{ dimrepeat('MU{}') }}))

    cdef int num_threads = pyiga.get_max_threads()

    for mu0 in prange(MU0, num_threads=num_threads, nogil=True):
        _asm_core_{{DIM}}d_kernel(asm, symmetric,
            {{ dimrepeat('bidx{}') }},
            {{ dimrepeat('transp{}') }},
            entries,
            mu0)
    return entries

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
cdef void _asm_core_{{DIM}}d_kernel(
    BaseAssembler{{DIM}}D asm,
    bint symmetric,
    {{ dimrepeat('unsigned[:, ::1] bidx{}') }},
    {{ dimrepeat('size_t[::1] transp{}') }},
    double[{{ dimrepeat(':') }}:1] entries,
    long _mu0
) nogil:
    cdef size_t[{{DIM}}] i, j
    cdef int {{ dimrepeat('diag{}') }}
    cdef double entry
    cdef long {{ dimrepeat('mu{}') }}, {{ dimrepeat('MU{}') }}

    mu0 = _mu0
    {{ dimrepeat('MU{}') }} = {{ dimrepeat('bidx{}.shape[0]') }}

    i[0] = bidx0[mu0, 0]
    j[0] = bidx0[mu0, 1]

    if symmetric:
        diag0 = <int>j[0] - <int>i[0]
        if diag0 > 0:       # block is above diagonal?
            return
{% for k in range(1, DIM) %}
{{ indent(k)   }}for mu{{k}} in range(MU{{k}}):
{{ indent(k)   }}    i[{{k}}] = bidx{{k}}[mu{{k}}, 0]
{{ indent(k)   }}    j[{{k}}] = bidx{{k}}[mu{{k}}, 1]

{{ indent(k)   }}    if symmetric:
{{ indent(k)   }}        diag{{k}} = <int>j[{{k}}] - <int>i[{{k}}]
{{ indent(k)   }}        if {{ dimrepeat('diag{} == 0', sep=' and ', upper=k) }} and diag{{k}} > 0:
{{ indent(k)   }}            continue
{% endfor %}
{{ indent(DIM) }}entry = asm.assemble_impl(i, j)
{{ indent(DIM) }}entries[{{ dimrepeat('mu{}') }}] = entry

{{ indent(DIM) }}if symmetric:
{{ indent(DIM) }}    if {{ dimrepeat('diag{} != 0', sep=' or ') }}:     # are we off the diagonal?
{{ indent(DIM) }}        entries[ {{ dimrepeat('transp{0}[mu{0}]') }} ] = entry   # then also write into the transposed entry


cdef generic_assemble_{{DIM}}d_parallel(BaseAssembler{{DIM}}D asm, symmetric=False):
    mlb = MLBandedMatrix(
        tuple(asm.ndofs),
        tuple(asm.p)
    )
    X = generic_assemble_core_{{DIM}}d(asm, mlb.bidx, symmetric=symmetric)
    mlb.data = X
    return mlb.asmatrix()


# helper function for fast low-rank assembler
cdef double _entry_func_{{DIM}}d(size_t i, size_t j, void * data):
    return (<BaseAssembler{{DIM}}D>data).assemble(i, j)



cdef class BaseVectorAssembler{{DIM}}D:
    cdef int nqp
    cdef size_t[{{DIM}}] ndofs
    cdef vector[ssize_t[:,::1]] meshsupp

    cdef void base_init(self, kvs):
        assert len(kvs) == {{DIM}}, "Assembler requires two knot vectors"
        self.nqp = max([kv.p for kv in kvs]) + 1
        self.ndofs[:] = [kv.numdofs for kv in kvs]
        self.meshsupp = [kvs[k].mesh_support_idx_all() for k in range({{DIM}})]

    cdef BaseAssembler{{DIM}}D shared_clone(self):
        return self     # by default assume thread safety

    cdef inline size_t to_seq(self, size_t[{{DIM + 1}}] ii) nogil:
        return {{ to_seq(indices + [ 'ii[%d]' % DIM ], ndofs + [ DIM|string ]) }}

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef inline void from_seq(self, size_t i, size_t[{{DIM + 1}}] out) nogil:
        out[{{DIM}}] = i % {{DIM}}
        i /= {{DIM}}
        {%- for k in range(1, DIM)|reverse %}
        out[{{k}}] = i % self.ndofs[{{k}}]
        i /= self.ndofs[{{k}}]
        {%- endfor %}
        out[0] = i

    cdef void assemble_impl(self, size_t[{{DIM}}] i, size_t[{{DIM}}] j, double result[]) nogil:
        pass


@cython.boundscheck(False)
@cython.wraparound(False)
cdef object generic_assemble_core_vec_{{DIM}}d(BaseVectorAssembler{{DIM}}D asm, bidx, bint symmetric=False):
    cdef unsigned[:, ::1] {{ dimrepeat('bidx{}') }}
    cdef long {{ dimrepeat('mu{}') }}, {{ dimrepeat('MU{}') }}
    cdef double[{{ dimrepeat(':') }}, ::1] entries

    {{ dimrepeat('bidx{}') }} = bidx
    {{ dimrepeat('MU{}') }} = {{ dimrepeat('bidx{}.shape[0]') }}

    cdef size_t[::1] {{ dimrepeat('transp{}') }}
    if symmetric:
    {%- for k in range(DIM) %}
        transp{{k}} = get_transpose_idx_for_bidx(bidx{{k}})
    {%- endfor %}
    else:
        {{ dimrepeat('transp{}', sep=' = ') }} = None

    entries = np.zeros(({{ dimrepeat('MU{}') }}, {{DIM * DIM}}))

    cdef int num_threads = pyiga.get_max_threads()

    for mu0 in prange(MU0, num_threads=num_threads, nogil=True):
        _asm_core_vec_{{DIM}}d_kernel(asm, symmetric,
            {{ dimrepeat('bidx{}') }},
            {{ dimrepeat('transp{}') }},
            entries,
            mu0)
    return entries

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
cdef void _asm_core_vec_{{DIM}}d_kernel(
    BaseVectorAssembler{{DIM}}D asm,
    bint symmetric,
    {{ dimrepeat('unsigned[:, ::1] bidx{}') }},
    {{ dimrepeat('size_t[::1] transp{}') }},
    double[{{ dimrepeat(':') }}, ::1] entries,
    long _mu0
) nogil:
    cdef size_t[{{DIM}}] i, j
    cdef int {{ dimrepeat('diag{}') }}
    cdef long {{ dimrepeat('mu{}') }}, {{ dimrepeat('MU{}') }}
    cdef int row, col

    mu0 = _mu0
    {{ dimrepeat('MU{}') }} = {{ dimrepeat('bidx{}.shape[0]') }}

    i[0] = bidx0[mu0, 0]
    j[0] = bidx0[mu0, 1]

    if symmetric:
        diag0 = <int>j[0] - <int>i[0]
        if diag0 > 0:       # block is above diagonal?
            return
{% for k in range(1, DIM) %}
{{ indent(k)   }}for mu{{k}} in range(MU{{k}}):
{{ indent(k)   }}    i[{{k}}] = bidx{{k}}[mu{{k}}, 0]
{{ indent(k)   }}    j[{{k}}] = bidx{{k}}[mu{{k}}, 1]

{{ indent(k)   }}    if symmetric:
{{ indent(k)   }}        diag{{k}} = <int>j[{{k}}] - <int>i[{{k}}]
{{ indent(k)   }}        if {{ dimrepeat('diag{} == 0', sep=' and ', upper=k) }} and diag{{k}} > 0:
{{ indent(k)   }}            continue
{% endfor %}
{{ indent(DIM) }}asm.assemble_impl(i, j, &entries[ {{ dimrepeat('mu{}') }}, 0 ])

{{ indent(DIM) }}if symmetric:
{{ indent(DIM) }}    if {{ dimrepeat('diag{} != 0', sep=' or ') }}:     # are we off the diagonal?
{{ indent(DIM) }}        for row in range({{DIM}}):
{{ indent(DIM) }}            for col in range({{DIM}}):
{{ indent(DIM) }}                entries[{{ dimrepeat('transp{0}[mu{0}]') }}, col*{{DIM}} + row] = entries[{{ dimrepeat('mu{}') }}, row*{{DIM}} + col]

''')


################################################################################
# Expressions for use in assembler generators
################################################################################

class Expr:
    def __add__(self, other): return OperExpr('+', self, other)
    def __sub__(self, other): return OperExpr('-', self, other)
    def __mul__(self, other): return OperExpr('*', self, other)
    def __div__(self, other):     return OperExpr('/', self, other)
    def __truediv__(self, other): return OperExpr('/', self, other)

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
        return reduce(operator.or_, (x.depends() for x in self.children), set())

    def is_named_expr(self):
        return any(isinstance(self, T)
                for T in (NamedScalarExpr, NamedVectorExpr, NamedMatrixExpr))

    def dx(self, k, times=1):
        return Dx(self, k, times)

    def dt(self, times=1):
        return Dt(self, times)

class VectorExpr(Expr):
    def dot(self, x):
        assert x.is_vector(), 'only dot(vec, vec) implemented'
        return inner(self, x)

class MatrixExpr(Expr):
    @property
    def T(self):
        return TransposedMatrixExpr(self)
    def dot(self, x):
        return matmul(self, x)

def named_expr(var, shape=(), symmetric=False):
    if shape is ():
        return NamedScalarExpr(var)
    elif len(shape) == 1:
        return NamedVectorExpr(var, shape)
    elif len(shape) == 2:
        return NamedMatrixExpr(var, shape, symmetric=symmetric)
    else:
        assert False, 'invalid shape'

class NamedScalarExpr(Expr):
    def __init__(self, var):
        self.var = var
        self.shape = ()
        self.children = ()

    def gencode(self):
        return self.var.name
    def depends(self):
        return set((self.var.name,))

class NamedVectorExpr(VectorExpr):
    def __init__(self, var, shape=()):
        self.var = var
        assert len(shape) == 1
        self.shape = shape
        self.children = ()
    def depends(self):
        return set((self.var.name,))

    def __getitem__(self, i):
        if isinstance(i, slice) or isinstance(i, range):
            return SliceExpr(self, i)
        else:
            return IndexExpr(self, i)

class NamedMatrixExpr(MatrixExpr):
    """Matrix expression which is represented by a matrix reference and shape."""
    def __init__(self, var, shape, symmetric=False):
        self.var = var
        self.shape = tuple(shape)
        assert len(self.shape) == 2
        self.symmetric = symmetric
        self.children = ()
    def depends(self):
        return set((self.var.name,))

    def to_seq(self, i, j):
        if self.symmetric and i > j:
            i, j = j, i
        return i * self.shape[0] + j

    def __getitem__(self, ij):
        i = _to_indices(ij[0], self.shape[0])
        j = _to_indices(ij[1], self.shape[1])
        if np.isscalar(i) and np.isscalar(j):
            return MatrixEntryExpr(self, i, j)
        else:
            return LiteralMatrixExpr([
                    [self[ii,jj] for jj in j]
                    for ii in i])

class LiteralVectorExpr(VectorExpr):
    """Vector expression which is represented by a list of individual expressions."""
    def __init__(self, entries):
        entries = tuple(entries)
        self.shape = (len(entries),)
        self.children = entries
        if not all(isinstance(e, Expr) and e.is_scalar() for e in self.children):
            raise ValueError('all vector entries should be scalar expressions')

    def __getitem__(self, i):
        return self.children[i]

class LiteralMatrixExpr(MatrixExpr):
    """Matrix expression which is represented by a 2D array of individual expressions."""
    def __init__(self, entries):
        entries = np.array(entries, dtype=object)
        self.shape = entries.shape
        self.children = tuple(entries.flat)
        if not all(isinstance(e, Expr) and e.is_scalar() for e in self.children):
            raise ValueError('all matrix entries should be scalar expressions')

    def at(self, i, j):
        return self.children[i * self.shape[1] + j]

    def __getitem__(self, ij):
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

class MatrixEntryExpr(Expr):
    def __init__(self, mat, i, j):
        self.shape = ()
        self.i = i
        self.j = j
        self.children = (mat,)
    def gencode(self):
        return '{name}[{k}]'.format(name=self.x.var.name,
                k=self.x.to_seq(self.i, self.j))

class TransposedMatrixExpr(MatrixExpr):
    def __init__(self, mat):
        assert mat.is_matrix(), 'can only transpose matrices'
        self.shape = (mat.shape[1], mat.shape[0])
        self.children = (mat,)
    def __getitem__(self, ij):
        result = self.x[ij[1], ij[0]]
        return result.T if result.is_matrix() else result

class BroadcastToVectorExpr(VectorExpr):
    def __init__(self, expr, shape):
        self.shape = shape
        self.children = (expr,)
    def __getitem__(self, i):
        return self.x

class BroadcastToMatrixExpr(MatrixExpr):
    def __init__(self, expr, shape):
        self.shape = shape
        self.children = (expr,)
    def __getitem__(self, ij):
        return self.x

def OperExpr(oper, x, y):
    if x.is_scalar() and y.is_scalar():
        return ScalarOperExpr(oper, x, y)
    elif x.is_vector() and y.is_vector():
        return VectorOperExpr(oper, x, y)
    elif x.is_matrix() and y.is_matrix():
        return MatrixOperExpr(oper, x, y)
    elif x.is_scalar() and y.is_vector():
        return OperExpr(oper, BroadcastToVectorExpr(x, y.shape), y)
    elif x.is_scalar() and y.is_matrix():
        return OperExpr(oper, BroadcastToMatrixExpr(x, y.shape), y)
    else:
        raise TypeError('operation not implemented for shapes: {} {} {}'.format(oper, x.shape, y.shape))

class ScalarOperExpr(Expr):
    def __init__(self, oper, x, y):
        assert x.is_scalar() and y.is_scalar(), 'expected scalars'
        assert x.shape == y.shape
        self.shape = x.shape
        self.oper = oper
        self.children = (x,y)

    def gencode(self):
        sep = ' ' + self.oper + ' '
        return '(' + sep.join(x.gencode() for x in self.children) + ')'

_oper_to_func = {
        '+': operator.add,
        '-': operator.sub,
        '*': operator.mul,
        '/': operator.truediv,
}

class VectorOperExpr(VectorExpr):
    def __init__(self, oper, x, y):
        assert x.is_vector() and y.is_vector(), 'expected vectors'
        assert x.shape == y.shape, 'incompatible shapes'
        self.shape = x.shape
        self.oper = oper
        self.children = (x,y)

    def __getitem__(self, i):
        func = _oper_to_func[self.oper]
        return reduce(func, (z[i] for z in self.children))

class MatrixOperExpr(MatrixExpr):
    def __init__(self, oper, x, y):
        assert x.is_matrix() and y.is_matrix(), 'expected matrices'
        assert x.shape == y.shape, 'incompatible shapes'
        self.shape = x.shape
        self.oper = oper
        self.children = (x,y)

    def __getitem__(self, ij):
        func = _oper_to_func[self.oper]
        return reduce(func, (z[ij] for z in self.children))

class IndexExpr(Expr):
    def __init__(self, x, i):
        assert isinstance(x, NamedVectorExpr)   # can only index named vectors
        self.shape = ()
        assert x.is_vector(), 'indexed expression is not a vector'
        i = int(i)
        if i < 0: i += x.shape[0]
        assert 0 <= i < x.shape[0], 'index out of range'
        self.i = i
        self.children = (x,)

    def gencode(self):
        return '{x}[{i}]'.format(x=self.x.var.name, i=self.i)

class PartialDerivExpr(Expr):
    """A scalar expression which refers to the value of a basis function or one
    of its partial derivatives."""
    def __init__(self, basisfun, D):
        self.shape = ()
        self.basisfun = basisfun
        self.D = tuple(D)
        self.children = ()

    def gencode(self):
        return self.basisfun.asmgen.gen_pderiv(self.basisfun, self.D)
    def depends(self):
        return set(('@' + self.basisfun.name,))

    def dx(self, k, times=1):
        Dnew = list(self.D)
        Dnew[k] += times
        return PartialDerivExpr(self.basisfun, Dnew)

def Dx(expr, k, times=1):
    if expr.is_named_expr():
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
    if expr.is_named_expr():
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
    if expr.is_named_expr():
        expr = expr.var.expr    # access underlying expression - mild hack
    if not isinstance(expr, PartialDerivExpr):
        raise TypeError('can only compute gradient of basis function')
    if dims is None:
        dims = expr.basisfun.vform.spacedims
    return LiteralVectorExpr(Dx(expr, k) for k in dims)


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


class SliceExpr(VectorExpr):
    def __init__(self, x, sl):
        assert x.is_vector(), 'expression is not a vector'
        if isinstance(sl, slice):
            self.indices = _indices_from_slice(sl, x.shape[0])
        else:   # range?
            self.indices = tuple(sl)
        for i in self.indices:
            assert 0 <= i < x.shape[0], 'slice index out of range'
        self.shape = (len(self.indices),)
        self.children = (x,)

    def __getitem__(self, i):
        assert not isinstance(i, slice), 'slicing of slices not implemented'
        return self.x[self.indices[i]]

class MatVecExpr(VectorExpr):
    def __init__(self, A, x):
        assert A.is_matrix() and x.is_vector()
        assert A.shape[1] == x.shape[0], 'incompatible shapes'
        self.shape = (A.shape[0],)
        self.children = (A, x)

    def __getitem__(self, i):
        if i < 0: i += self.shape[0]
        assert 0 <= i < self.shape[0]
        return reduce(operator.add,
            (self.x[i, j] * self.y[j] for j in range(self.y.shape[0])))

class MatMatExpr(MatrixExpr):
    def __init__(self, A, B):
        assert A.is_matrix() and B.is_matrix()
        assert A.shape[1] == B.shape[0], 'incompatible shapes'
        self.shape = (A.shape[0], B.shape[1])
        self.children = (A, B)

    def __getitem__(self, ij):
        assert len(ij) == 2
        i, j = ij
        if i < 0: i += self.shape[0]
        assert 0 <= i < self.shape[0]
        if j < 0: j += self.shape[1]
        assert 0 <= j < self.shape[1]
        return reduce(operator.add,
            (self.x[i, k] * self.y[k, j] for k in range(self.x.shape[1])))

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

def as_vector(x): return LiteralVectorExpr(x)
def as_matrix(x): return LiteralMatrixExpr(x)

def matmul(a, b):
    if a.is_matrix() and b.is_vector():
        return MatVecExpr(a, b)
    elif a.is_matrix() and b.is_matrix():
        return MatMatExpr(a, b)
    else:
        raise TypeError('invalid types in matmul')


################################################################################
# concrete assembler generators
################################################################################

def mass_vf(dim):
    V = VForm(dim)
    V.add(V.W * V.u * V.v)
    return V


def stiffness_vf(dim):
    V = VForm(dim)
    B = V.let('B', V.W * matmul(V.JacInv, V.JacInv.T), symmetric=True)
    V.add(B.dot(V.gu).dot(V.gv))
    return V

## slower:
#def stiffness_vf(dim):
#    V = VForm(dim)
#    V.add(V.W * inner(V.gradu, V.gradv))
#    return V


def heat_st_vf(dim):
    V = VForm(dim, spacetime=True)
    ut_v = V.u.dt() * V.v
    gradgrad = inner(V.gradu, V.gradv)
    V.add(V.W * (gradgrad + ut_v))
    return V


def wave_st_vf(dim):
    V = VForm(dim, spacetime=True)

    utt_vt = V.u.dt(2) * V.v.dt()
    dtgv = V.let('dtgv', grad(V.v).dt()) # time derivative of gradient (parametric coordinates)
    dtgradv = V.let('dtgradv', matmul(V.JacInv_x.T, dtgv))  # transform space gradient
    gradu_dtgradv = inner(V.gradu, dtgradv)

    V.add(V.W * (utt_vt + gradu_dtgradv))
    return V


def divdiv_vf(dim):
    V = VForm(dim, vec=dim**2)
    for i in range(dim):
        for j in range(dim):
            V.add_at(dim*i + j, V.W * V.gradu[j] * V.gradv[i])
    return V


################################################################################
# Assembler generation script, main entry point
################################################################################

def generate(dim):
    DIM = dim

    def dimrepeat(s, sep=', ', upper=DIM):
        return sep.join([s.format(k) for k in range(upper)])

    def to_seq(i, n):
        s = i[0]
        for k in range(1, len(i)):
            s = '({0}) * {1} + {2}'.format(s, n[k], i[k])
        return s

    def indent(num):
        return num * '    ';

    # helper variables for to_seq
    indices = ['ii[%d]' % k for k in range(DIM)]
    ndofs   = ['self.ndofs[%d]' % k for k in range(DIM)]

    code = PyCode()

    def gen(vf, classname):
        AsmGenerator(vf, classname, code).generate()

    gen(mass_vf(DIM), 'MassAssembler')
    gen(stiffness_vf(DIM), 'StiffnessAssembler')
    gen(heat_st_vf(DIM), 'HeatAssembler_ST')
    gen(wave_st_vf(DIM), 'WaveAssembler_ST')
    gen(divdiv_vf(DIM), 'DivDivAssembler')

    s = tmpl_generic.render(locals())
    s += '\n\n'
    s += code.result()
    return s

if __name__ == '__main__':
    path = os.path.join(os.path.dirname(__file__), "..", "pyiga", "assemblers.pxi")
    with open(path, 'w') as f:
        f.write('# file generated by generate-assemblers.py\n')
        f.write(generate(dim=2))
        f.write(generate(dim=3))

