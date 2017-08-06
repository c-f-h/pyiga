import os.path
from jinja2 import Template
from collections import OrderedDict, defaultdict
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
        # predefined local variables with their generators (created on demand)
        self.predefined_vars = {
            'JacInv': lambda self: inv(self.Jac),
            'W':      lambda self: self.GaussWeight * abs(det(self.Jac)),
        }

    def basisfuns(self, parametric=False):
        return tuple(self.basisval(bf, physical=not parametric) for bf in self.basis_funs)

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
            if expr.is_matrix():
                expr = expr.ravel()
            if not expr.shape == (self.vec,):
                raise TypeError('vector assembler requires vector or matrix expression of proper length')
        else:
            if not expr.is_scalar():
                raise TypeError('require scalar expression')
        self.exprs.append(expr)

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
        def applyfun(e):
            e2 = None
            if type is None or isinstance(e, type):
                e2 = fun(e)
            return e2 if e2 is not None else e
        self.exprs = mapexprs(self.exprs, applyfun, deep=True)

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
        # find common subexpressions and extract them into named variables
        self.extract_common_expressions()
        # perform dependency analysis for expressions and variables
        self.dependency_analysis()

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

    def start_loop_with_fields(self, fields_in, fields_out=[], local_vars=[]):
        fields = fields_in + fields_out

        # get input size from an arbitrary field variable
        for k in range(self.dim):
            self.declare_index(
                    'n%d' % k,
                    '_{var}.shape[{k}]'.format(k=k, var=fields[0].name)
            )

        # temp storage for local variables
        for var in local_vars:
            self.declare_var(var)

        # temp storage for field variables
        for var in fields:
            self.declare_var(var, ref=True)

        # declare iteration indices
        for k in range(self.dim):
            self.declare_index('i%d' % k)

        # start the for loop
        self.put('')
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

        # generate code for computing local variables
        for var in local_vars:
            self.gen_assign(var, var.expr)

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

        self.declare_custom_variables()

        self.put('')

        ############################################################
        # main loop over all Gauss points
        self.start_loop_with_fields(field_params, local_vars=local_vars)

        # if needed, generate custom code for the bilinear form a(u,v)
        self.generate_biform_custom()

        # generate code for all expressions in the bilinear form
        if self.vec:
            for expr in self.vform.exprs:
                for i, e_i in enumerate(expr):
                    self.put(('result[%d] += ' % i) + e_i.gencode())
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

        # compute Jacobian
        if 'Jac' in vf.vars:
            self.put('geo_jac = geo.grid_jacobian(gaussgrid)')

        if 'GaussWeight' in vf.vars:
            self.put('gauss_weights = %s' % self.tensorprod('gaussweights'))

        for var in vf.field_vars():
            if var.src:
                self.putf("self.{name} = {src}", name=var.name, src=var.src)
            elif var.expr:  # custom precomputed field var
                self.putf("self.{name} = np.empty(N + {shape})", name=var.name, shape=var.shape)

        if vf.precomp:
            # call precompute function
            self.putf('{classname}{dim}D.precompute_fields(', classname=self.classname)
            self.indent(2)
            for var in vf.precomp_deps: # input fields
                self.put(var.src + ',')
            for var in vf.precomp:      # output fields
                self.put('self.' + var.name + ',')
            self.dedent(2)
            self.put(')')

        self.initialize_custom_fields()
        self.end_function()

    def generate_precomp(self):
        vf = self.vform

        # function header
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

        # start main loop
        self.start_loop_with_fields(vf.precomp_deps, fields_out=vf.precomp, local_vars=vf.precomp_locals)

        # generate assignment statements
        I = self.dimrep('i{}')  # current grid index
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
        self.vform.finalize()
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
    base_complexity = 0

class VectorVarExpr(VarExpr):
    def __init__(self, var):
        self.var = var
        self.shape = var.shape
        assert len(self.shape) == 1
        self.children = ()

    def at(self, i):
        return VectorEntryExpr(self, i)
    base_complexity = 0

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
    base_complexity = 0

class LiteralVectorExpr(Expr):
    """Vector expression which is represented by a list of individual expressions."""
    def __init__(self, entries):
        entries = tuple(entries)
        self.shape = (len(entries),)
        self.children = entries
        if not all(isinstance(e, Expr) and e.is_scalar() for e in self.children):
            raise ValueError('all vector entries should be scalar expressions')
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
        self.children = tuple(entries.flat)
        if not all(isinstance(e, Expr) and e.is_scalar() for e in self.children):
            raise ValueError('all matrix entries should be scalar expressions')
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

    base_complexity = 0

    def __str__(self):
        return '%s[%i]' % (self.x.var.name, self.i)

    def hash_key(self):
        return (self.i,)

    def gencode(self):
        return '{x}[{i}]'.format(x=self.x.var.name, i=self.i)

class MatrixEntryExpr(Expr):
    def __init__(self, mat, i, j):
        assert isinstance(mat, MatrixVarExpr)
        self.shape = ()
        self.i = i
        self.j = j
        self.children = (mat,)

    base_complexity = 0

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

class TransposedMatrixExpr(Expr):
    def __init__(self, mat):
        if not mat.is_matrix(): raise TypeError('can only transpose matrices')
        self.shape = (mat.shape[1], mat.shape[0])
        self.children = (mat,)
    def __str__(self):
        return 'transpose(%s)' % self.x
    def __getitem__(self, ij):
        result = self.x[ij[1], ij[0]]
        return result.T if result.is_matrix() else result

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
    if oper == '+' and isinstance(y, NegExpr):
        return OperExpr('-', x, y.x)
    elif oper == '-' and isinstance(y, NegExpr):
        return OperExpr('+', x, y.x)
    elif x.is_scalar() and y.is_scalar():
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
        return '%s_%s (%s)' % (
                self.basisfun.name,
                ''.join(str(k) for k in self.D),
                'phys' if self.physical else 'para')

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
        assert A.shape[1] == x.shape[0], 'incompatible shapes'
        self.shape = (A.shape[0],)
        self.children = (A, x)

    def at(self, i):
        return reduce(operator.add,
            (self.x[i, j] * self.y[j] for j in range(self.y.shape[0])))

class MatMatExpr(Expr):
    def __init__(self, A, B):
        assert A.is_matrix() and B.is_matrix()
        assert A.shape[1] == B.shape[0], 'incompatible shapes'
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
    if not isinstance(expr, PartialDerivExpr):
        raise TypeError('can only compute gradient of basis function')
    if dims is None:
        dims = expr.basisfun.vform.spacedims
    return LiteralVectorExpr(Dx(expr, k) for k in dims)

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

def _pm(k, a):
    if k % 2: return -a
    else:     return a

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
                (_pm(j, A[0,j] * minor(A, 0, j)) for j in range(n)))

def inv(A):
    """Inverse of a matrix.."""
    if not A.is_matrix() or A.shape[0] != A.shape[1]:
        raise ValueError('can only compute inverse of square matrices')
    n = A.shape[0]
    invdet = ConstExpr(1) / det(A)
    cofacs = as_matrix(
            [[ _pm(i+j, minor(A, i, j)) for i in range(n)]
                for j in range(n)])
    return invdet * cofacs

def cross(x, y):
    return VectorCrossExpr(x, y)

def outer(x, y):
    return OuterProdExpr(x, y)


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

################################################################################
# concrete assembler generators
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
    u, v = V.basisfuns()
    V.add(outer(grad(v), grad(u)) * dx)
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

