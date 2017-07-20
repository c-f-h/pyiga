import os.path
from jinja2 import Template
from collections import OrderedDict
from functools import reduce
import operator
import numpy as np

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
        else:
            self.src = src
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

class AsmGenerator:
    def __init__(self, classname, code, dim, numderiv=0, vec=False):
        self.classname = classname
        self.code = code
        self.dim = dim
        self.vec = vec
        self.numderiv = numderiv
        self.vars = OrderedDict()
        self.exprs = []         # expressions to be added to the result
        # variables provided during initialization (geo_XXX)
        self.init_det = False
        self.init_jac = False
        self.init_jacinv = False
        self.init_weights = False       # geo_weights = Gauss weights * abs(geo_det)
        # predefined local variables with their generators (created on demand)
        self.predefined_vars = {
            'u':     lambda self: self.basisval('u'),
            'v':     lambda self: self.basisval('v'),
            'gu':    lambda self: self.gradient('u'),
            'gv':    lambda self: self.gradient('v'),
            'gradu': lambda self: matmul(self.JacInv.T, self.gu),
            'gradv': lambda self: matmul(self.JacInv.T, self.gv),
        }
        self.cached_vars = {}

    def local_vars(self):
        return (var for var in self.vars.values() if var.is_local())
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

    def add(self, expr):
        assert not self.vec, 'add() is only for scalar assemblers; use add_at()'
        assert expr.is_scalar(), 'require scalar expression'
        self.exprs.append(expr)

    def add_at(self, idx, expr):
        assert self.vec, 'add_at() is only for vector assemblers; use add()'
        assert expr.is_scalar(), 'require scalar expression'
        self.exprs.append((idx, expr))

    def partial_deriv(self, var, D):
        """Parametric partial derivative of `var` of order `D=(Dx1, ..., Dxd)`"""
        self.numderiv = max(max(D), self.numderiv)    # make sure derivatives are computed
        return PartialDerivExpr(var, D, self)

    def basisval(self, var):
        return self.partial_deriv(var, self.dim * (0,))

    def gradient(self, var, dims=None, additional_derivs=None):
        if dims is None:
            dims = range(self.dim)
        if additional_derivs is None:
            additional_derivs = self.dim * [0]

        entries = []
        for k in dims:
            D = list(additional_derivs)
            D[k] += 1
            entries.append(self.partial_deriv(var, D))
        return LiteralVectorExpr(entries)

    def let(self, varname, expr):
        self.vars[varname] = AsmVar(varname, expr, shape=None, local=True)
        return named_expr(varname, shape=expr.shape)

    # hooks for custom code generation

    def declare_temp_variables(self):
        pass

    def initialize_fields(self):
        pass

    def generate_biform(self):
        pass

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
            self.register_scalar_field('W', src='geo_weights')
            self.cached_vars['W'] = named_expr('W')
        return self.cached_vars['W']

    @property
    def JacInv(self):
        if not 'JacInv' in self.cached_vars:
            self.init_jacinv = True
            self.cached_vars['JacInv'] = self.register_matrix_field('JacInv', shape=(self.dim,self.dim), src='geo_jacinv')
        return self.cached_vars['JacInv']

    ##################################################
    # code generation
    ##################################################

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

    def gen_pderiv(self, var, D, idx='i'):
        """Generate code for computing parametric partial derivative of `var` of order `D=(Dx1, ..., Dxd)`"""
        D = tuple(reversed(D))  # x is last axis
        assert len(D) == self.dim
        assert all(0 <= d <= self.numderiv for d in D)
        factors = [
                "VD{var}{k}[{nderiv}*{idx}{k}+{ofs}]".format(
                    var = var,
                    idx = idx,
                    k   = k,
                    ofs = D[k],
                    **self.env
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

    def gen_assign(self, out, expr):
        if expr.is_vector():
            for k in range(expr.shape[0]):
                self.putf('{name}[{k}] = {rhs}',
                        name=out,
                        k=k,
                        rhs=expr[k].gencode())
        elif expr.is_matrix():
            m, n = expr.shape
            for i in range(m):
                for j in range(n):
                    # TODO: symmetry optimization
                    self.putf('{name}[{k}] = {rhs}',
                            name=out,
                            k=i*m + j,
                            rhs=expr[i,j].gencode())
        else:
            self.put(out + ' = ' + expr.gencode())

    def cython_pragmas(self):
        self.put('@cython.boundscheck(False)')
        self.put('@cython.wraparound(False)')
        self.put('@cython.initializedcheck(False)')

    def generate_kernel(self):
        # function definition
        self.cython_pragmas()
        self.put('@staticmethod')
        rettype = 'void' if self.vec else 'double'
        self.putf('cdef {rettype} combine(', rettype=rettype)
        self.indent(2)

        # parameters
        for var in self.field_vars():
            self.putf('double[{X}:1] _{name},',
                    X=', '.join((self.dim + len(var.shape)) * ':'),
                    name=var.name)

        self.put(self.dimrep('double* VDu{}') + ',')
        self.put(self.dimrep('double* VDv{}') + ',')
        if self.vec:    # for vector assemblers, result is passed as a pointer
            self.put('double result[]')
        self.dedent()
        self.put(') nogil:')

        # get input size from an arbitrary field variable
        first_var = list(self.field_vars())[0]
        for k in range(self.dim):
            self.declare_index(
                    'n%d' % k,
                    '_{var}.shape[{k}]'.format(k=k, var=first_var.name)
            )
        self.put('')

        # local variables
        for k in range(self.dim):
            self.declare_index('i%d' % k)

        if not self.vec:    # for vector assemblers, result is passed as a pointer
            self.declare_scalar('result', '0.0')

        # temp storage for field variables
        for var in self.field_vars():
            if var.is_scalar():
                self.declare_scalar(var.name)
            elif var.is_vector() or var.is_matrix():
                self.declare_pointer(var.name)

        # temp storage for local variables
        for var in self.local_vars():
            if var.is_vector():
                self.declare_vec(var.name, size=var.shape[0])
            elif var.is_matrix():
                self.declare_vec(var.name, size=var.shape[0]*var.shape[1])
            else:
                self.declare_scalar(var.name)

        self.declare_temp_variables()

        self.put('')

        # main loop over all Gauss points
        for k in range(self.dim):
            self.code.for_loop('i%d' % k, 'n%d' % k)

        I = self.dimrep('i{}')

        # generate assignments for field variables
        for var in self.field_vars():
            if var.is_scalar():
                self.putf('{name} = _{name}[{I}]', name=var.name, I=I)
            elif var.is_matrix():
                self.putf('{name} = &_{name}[{I}, 0, 0]', name=var.name, I=I)
        self.put('')

        # generate assignment statements for local variables
        for var in self.local_vars():
            self.gen_assign(var.name, var.expr)

        # if needed, generate custom code for the bilinear form a(u,v)
        self.generate_biform()

        # generate code for all expressions in the bilinear form
        if self.vec:
            for idx, expr in self.exprs:
                self.put(('result[%d] += ' % idx) + expr.gencode())
        else:
            for expr in self.exprs:
                self.put('result += ' + expr.gencode())

        # end main loop
        for _ in range(self.dim):
            self.code.end_loop()

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
        for var in self.field_vars():
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
        self.put('def __init__(self, kvs, geo):')
        self.indent()
        for line in \
"""assert geo.dim == {dim}, "Geometry has wrong dimension"
self.base_init(kvs)

gaussgrid, gaussweights = make_tensor_quadrature([kv.mesh for kv in kvs], self.nqp)

self.C = compute_values_derivs(kvs, gaussgrid, derivs={maxderiv})""".splitlines():
            self.putf(line)
        self.put('')

        # dependencies
        if self.init_weights: self.init_det = True
        if self.init_jacinv or self.init_det: self.init_jac = True

        # compute Jacobian
        if self.init_jac:
            self.put('geo_jac = geo.grid_jacobian(gaussgrid)')

        # determinant and/or inverse of Jacobian
        if self.init_det and self.init_jacinv:
            self.put('geo_det, geo_jacinv = det_and_inv(geo_jac)')
        elif self.init_det:
            self.put('geo_det = determinants(geo_jac)')
        elif self.init_jacinv:
            self.put('geo_jacinv = inverses(geo_jac)')

        if self.init_weights:
            self.put('geo_weights = %s * np.abs(geo_det)' % self.tensorprod('gaussweights'))

        for var in self.field_vars():
            if var.src:
                self.putf("self.{name} = {src}", name=var.name, src=var.src)

        self.initialize_fields()
        self.end_function()

    # main code generation entry point

    def generate(self):
        self.env = {
            'dim': self.dim,
            'nderiv': self.numderiv+1, # includes 0-th derivative
            'maxderiv': self.numderiv,
        }

        baseclass = 'BaseVectorAssembler' if self.vec else 'BaseAssembler'
        self.putf('cdef class {classname}{dim}D({base}{dim}D):',
                classname=self.classname, base=baseclass)
        self.indent()

        # declare instance variables
        self.put('cdef vector[double[:, :, ::1]] C       # 1D basis values. Indices: basis function, mesh point, derivative')
        # declare field variables
        for var in self.field_vars():
            self.putf('cdef double[{X}:1] {name}',
                    X=', '.join((self.dim + len(var.shape)) * ':'),
                    name=var.name)
        self.put('')

        # generate methods
        self.generate_init()
        self.put('')
        self.generate_kernel()
        self.put('')
        self.generate_assemble_impl()
        self.dedent()
        self.put('')


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
    def __div__(self, other): return OperExpr('/', self, other)
    def is_scalar(self):    return self.shape is ()
    def is_vector(self):    return len(self.shape) == 1
    def is_matrix(self):    return len(self.shape) == 2

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

def named_expr(s, shape=()):
    if shape is ():
        return NamedScalarExpr(s)
    elif len(shape) == 1:
        return NamedVectorExpr(s, shape)
    elif len(shape) == 2:
        return NamedMatrixExpr(s, shape)
    else:
        assert False, 'invalid shape'

class NamedScalarExpr(Expr):
    def __init__(self, s):
        self.s = s
        self.shape = ()

    def gencode(self):
        return self.s

class NamedVectorExpr(VectorExpr):
    def __init__(self, s, shape=()):
        self.s = s
        assert len(shape) == 1
        self.shape = shape

    def __getitem__(self, i):
        if isinstance(i, slice) or isinstance(i, range):
            return SliceExpr(self, i)
        else:
            return IndexExpr(self, i)

class LiteralVectorExpr(VectorExpr):
    """Vector expression which is represented by a list of individual expressions."""
    def __init__(self, entries):
        self.shape = (len(entries),)
        self.entries = entries

    def __getitem__(self, i):
        if i < 0: i += self.shape[0]
        assert 0 <= i < self.shape[0], 'index out of bounds'
        return self.entries[i]

class LiteralMatrixExpr(MatrixExpr):
    """Matrix expression which is represented by a 2D array of individual expressions."""
    def __init__(self, entries):
        self.entries = np.array(entries, dtype=object)
        self.shape = self.entries.shape

    def __getitem__(self, ij):
        i = _to_indices(ij[0], self.shape[0])
        j = _to_indices(ij[1], self.shape[1])
        if np.isscalar(i) and np.isscalar(j):
            return self.entries[i, j]
        else:
            return LiteralMatrixExpr(self.entries[np.ix_(i, j)])

class NamedMatrixExpr(MatrixExpr):
    """Matrix expression which is represented by a matrix reference and shape."""
    def __init__(self, name, shape, symmetric=False):
        self.name = name
        self.shape = tuple(shape)
        assert len(self.shape) == 2
        self.symmetric = symmetric

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

class MatrixEntryExpr(Expr):
    def __init__(self, mat, i, j):
        self.shape = ()
        self.mat = mat
        self.i = i
        self.j = j
    def gencode(self):
        return '{name}[{k}]'.format(name=self.mat.name,
                k=self.mat.to_seq(self.i, self.j))

class TransposedMatrixExpr(MatrixExpr):
    def __init__(self, mat):
        assert mat.is_matrix(), 'can only transpose matrices'
        self.mat = mat
        self.shape = (mat.shape[1], mat.shape[0])
    def __getitem__(self, ij):
        return self.mat[ij[1], ij[0]]

class OperExpr(Expr):
    def __init__(self, oper, x, y):
        assert x.is_scalar() and y.is_scalar(), 'expected scalars'
        assert x.shape == y.shape
        self.shape = x.shape
        self.oper = oper
        self.args = (x,y)

    def gencode(self):
        sep = ' ' + self.oper + ' '
        return '(' + sep.join(x.gencode() for x in self.args) + ')'

class IndexExpr(Expr):
    def __init__(self, x, i):
        assert isinstance(x, NamedVectorExpr)   # can only index named vectors
        self.shape = ()
        assert x.is_vector(), 'indexed expression is not a vector'
        self.x = x
        i = int(i)
        if i < 0: i += x.shape[0]
        assert 0 <= i < x.shape[0], 'index out of range'
        self.i = i

    def gencode(self):
        return '{x}[{i}]'.format(x=self.x.s, i=self.i)

class PartialDerivExpr(Expr):
    def __init__(self, var, D, asmgen):
        self.shape = ()
        self.var = var
        self.D = D
        self.asmgen = asmgen

    def gencode(self):
        return self.asmgen.gen_pderiv(self.var, self.D)

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
        return x
    else:
        return tuple(x)


class SliceExpr(VectorExpr):
    def __init__(self, x, sl):
        assert x.is_vector(), 'expression is not a vector'
        self.x = x
        if isinstance(sl, slice):
            self.indices = _indices_from_slice(sl, x.shape[0])
        else:   # range?
            self.indices = tuple(sl)
        for i in self.indices:
            assert 0 <= i < x.shape[0], 'slice index out of range'
        self.shape = (len(self.indices),)

    def __getitem__(self, i):
        assert not isinstance(i, slice), 'slicing of slices not implemented'
        return self.x[self.indices[i]]

class MatVecExpr(VectorExpr):
    def __init__(self, A, x):
        assert A.is_matrix() and x.is_vector()
        assert A.shape[1] == x.shape[0], 'incompatible shapes'
        self.shape = (A.shape[0],)
        self.A = A
        self.x = x

    def __getitem__(self, i):
        if i < 0: i += self.shape[0]
        assert 0 <= i < self.shape[0]
        return reduce(operator.add,
            (self.A[i, j] * self.x[j] for j in range(self.x.shape[0])))

class MatMatExpr(MatrixExpr):
    def __init__(self, A, B):
        assert A.is_matrix() and B.is_matrix()
        assert A.shape[1] == B.shape[0], 'incompatible shapes'
        self.shape = (A.shape[0], B.shape[1])
        self.A = A
        self.B = B

    def __getitem__(self, ij):
        assert len(ij) == 2
        i, j = ij
        if i < 0: i += self.shape[0]
        assert 0 <= i < self.shape[0]
        if j < 0: j += self.shape[1]
        assert 0 <= j < self.shape[1]
        return reduce(operator.add,
            (self.A[i, k] * self.B[k, j] for k in range(self.A.shape[1])))

def inner(x, y):
    assert x.is_vector() and y.is_vector(), 'inner() requires vector expressions'
    assert x.shape == y.shape, 'incompatible shapes'
    return reduce(operator.add, (x[i] * y[i] for i in range(x.shape[0])))

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

def MassAsmGen(code, dim):
    A = AsmGenerator('MassAssembler', code, dim)
    A.add(A.W * A.u * A.v)
    return A


class StiffnessAsmGen(AsmGenerator):
    def __init__(self, code, dim):
        AsmGenerator.__init__(self, 'StiffnessAssembler', code, dim)
        self.init_jacinv = self.init_weights = True

        B = self.register_matrix_field('B', symmetric=True)
        self.add(B.dot(self.gu).dot(self.gv))

    def initialize_fields(self):
        self.putf('self.B = matmatT(geo_jacinv) * geo_weights[{slices}, None, None]',
            slices=self.dimrep(':'))


## slower:
#def StiffnessAsmGen(code, dim):
#    A = AsmGenerator('StiffnessAssembler', code, dim)
#    A.add(A.W * inner(A.gradu, A.gradv))
#    return A

#def StiffnessAsmGen(code, dim):
#    A = AsmGenerator('StiffnessAssembler', code, dim)
#    B = A.let('B', matmul(A.JacInv, A.JacInv.T))
#    A.add(A.W * inner(matmul(B, A.gu), A.gv))
#    return A


def Heat_ST_AsmGen(code, dim):
    A = AsmGenerator('HeatAssembler_ST', code, dim)
    timederiv = A.gradu[-1] * A.v
    gradgrad = inner(A.gradu[:-1], A.gradv[:-1])
    A.add(A.W * (gradgrad + timederiv))
    return A


def Wave_ST_AsmGen(code, dim):
    A = AsmGenerator('WaveAssembler_ST', code, dim)

    spacedims, timedim = range(dim-1), dim-1
    Dt  = (dim-1) * (0,) + (1,)     # first time derivative
    Dtt = (dim-1) * (0,) + (2,)     # second time derivative

    utt_vt = A.partial_deriv('u', Dtt) * A.partial_deriv('v', Dt)

    # compute time derivative of gradient of v (assumes ST cylinder)
    dtgv = A.let('dtgv', A.gradient('v', dims=spacedims, additional_derivs=Dt))
    dtgradv = A.let('dtgradv', matmul(A.JacInv.T[spacedims, spacedims], dtgv))
    gradu_dtgradv = inner(A.gradu[spacedims], dtgradv)

    A.add(A.W * (utt_vt + gradu_dtgradv))
    return A


def DivDivAsmGen(code, dim):
    A = AsmGenerator('DivDivAssembler', code, dim, vec=dim**2)
    for i in range(dim):
        for j in range(dim):
            A.add_at(dim*i + j, A.W * A.gradu[j] * A.gradv[i])
    return A


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
    MassAsmGen(code, DIM).generate()
    StiffnessAsmGen(code, DIM).generate()
    Heat_ST_AsmGen(code, DIM).generate()
    Wave_ST_AsmGen(code, DIM).generate()
    DivDivAsmGen(code, DIM).generate()

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

