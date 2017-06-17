import os.path
from jinja2 import Template

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


class AsmGenerator:
    def __init__(self, classname, code, dim, numderiv):
        self.classname = classname
        self.code = code
        self.dim = dim
        self.env = {
            'dim': dim,
            'nderiv': numderiv+1, # includes 0-th derivative
            'maxderiv': numderiv,
        }
        self.vars = {}
        self.need_val = False
        self.need_grad = False
        self.need_det = False
        self.need_jac = False
        self.need_jacinv = False

    def register_scalar_field(self, name):
        self.vars[name] = {
            'type': 'scalar',
            'numdims': 0
        }

    def register_vector_field(self, name):
        self.vars[name] = {
            'type': 'vector',
            'numdims': 1
        }

    def register_matrix_field(self, name, symmetric=False):
        self.vars[name] = {
            'type': 'matrix',
            'numdims': 2,
            'symmetric': symmetric
        }

    def put(self, s):
        self.code.put(s)

    def putf(self, s, **kwargs):
        env = dict(self.env)
        env.update(kwargs)
        self.code.putf(s, **env)

    def indent(self, num=1):
        self.code.indent(num)

    def dedent(self, num=1):
        self.code.dedent(num)

    def dimrep(self, s, sep=', '):
        return sep.join([s.format(k, **self.env) for k in range(self.dim)])

    def extend_dim(self, i):
        # ex.: i = 1  ->  'None,:,None'
        slices = self.dim * ['None']
        slices[i] = ':'
        return ','.join(slices)

    def tensorprod(self, var):
        return ' * '.join(['{0}[{1}][{2}]'.format(var, k, self.extend_dim(k))
            for k in range(self.dim)])

    def grad_comp(self, var, idx, j):
        factors = [
                "{var}{k}[{nderiv}*{idx}{k}+{ofs}]".format(
                    var = var,
                    idx = idx,
                    k   = k,
                    ofs = (1 if k + j + 1 == self.dim else 0),
                    **self.env
                )
                for k in range(self.dim)]
        return ' * '.join(factors)

    def basisval(self, var, idx):
        factors = [
                "{var}{k}[{nderiv}*{idx}{k}+0]".format(
                    var = var,
                    idx = idx,
                    k   = k,
                    **self.env
                )
                for k in range(self.dim)]
        return ' * '.join(factors)

    def make_grad(self, result, var):
        for k in range(self.dim):
            self.putf('{res}[{k}] = {comp}',
                    res=result,
                    k=k,
                    comp=self.grad_comp(var, 'i', k))

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

    def mat_entry(self, name, i, j):
        sym = self.vars[name]['symmetric']
        if sym and i > j:
            i, j = j, i
        idx = i*self.dim + j
        return '{name}ptr[{idx}]'.format(name=name, idx=idx)

    def vec_entry(self, name, i):
        return '{name}[{i}]'.format(name=name, i=i)

    def matvec_comp(self, A, x, i):
        return ' + '.join(
                self.mat_entry(A, i, j) + '*' + self.vec_entry(x, j)
                for j in range(self.dim))

    def add_matvecvec(self, A, x, y):
        for k in range(self.dim):
            self.putf('result += ({Axk}) * {yk}',
                    Axk=self.matvec_comp(A, x, k),
                    yk =self.vec_entry(y, k))

    def generate_kernel(self):
        # function definition
        self.put('@cython.boundscheck(False)')
        self.put('@cython.wraparound(False)')
        self.put('@cython.initializedcheck(False)')
        self.put('@staticmethod')
        self.putf('cdef double combine(')
        self.indent(2)

        # parameters
        for name, var in self.vars.items():
            self.putf('double[{X}:1] _{name},',
                    X=', '.join((self.dim + var['numdims']) * ':'),
                    name=name)

        self.put(self.dimrep('double* VDu{}') + ',')
        self.put(self.dimrep('double* VDv{}') + ',')
        self.dedent()
        self.put(') nogil:')

        # get input size from an arbitrary field variable
        first_var = list(self.vars)[0]
        for k in range(self.dim):
            self.declare_index(
                    'n%d' % k,
                    '_{var}.shape[{k}]'.format(k=k, var=first_var)
            )
        self.put('')

        # local variables
        for k in range(self.dim):
            self.declare_index('i%d' % k)

        if self.need_val:
            self.declare_scalar('u')
            self.declare_scalar('v')
        if self.need_grad:
            self.declare_vec('gu')
            self.declare_vec('gv')

        self.declare_scalar('result', '0.0')

        # temp storage for field variables
        for name, var in self.vars.items():
            if var['type'] == 'scalar':
                self.declare_scalar(name)
            elif var['type'] == 'matrix':
                self.declare_pointer(name + 'ptr')

        self.put('')

        # main loop over all Gauss points
        for k in range(self.dim):
            self.code.for_loop('i%d' % k, 'n%d' % k)

        self.env['I'] = self.dimrep('i{}')

        for name, var in self.vars.items():
            if var['type'] == 'scalar':
                self.putf('{name} = _{name}[{I}]', name=name)
            elif var['type'] == 'matrix':
                self.putf('{name}ptr = &_{name}[{I}, 0, 0]', name=name)
        self.put('')

        if self.need_val:
            self.put('u = ' + self.basisval('VDu', 'i'))
            self.put('v = ' + self.basisval('VDv', 'i'))
            self.put('')
        if self.need_grad:
            self.make_grad('gu', 'VDu')
            self.put('')
            self.make_grad('gv', 'VDv')
            self.put('')

        # compute the bilinear form a(v,u)
        self.generate_biform()

        # end main loop
        for _ in range(self.dim):
            self.code.end_loop()

        self.put('return result')
        self.dedent()

    def generate_assemble_impl(self):
        tmpl = Template(macros + '    {{ assemble_impl_header() }}')
        src = tmpl.render({'DIM': self.dim})
        self.dedent()
        for line in src.splitlines():
            self.put(line)
        self.indent(2)

        self.put('')
        self.putf('return {classname}{dim}D.combine(', classname=self.classname)
        self.indent(2)
        for name in self.vars:
            self.putf('self.{name} [ {idx} ],', name=name,
                    idx=self.dimrep('g_sta[{0}]:g_end[{0}]'))
        self.put(self.dimrep('values_i[{0}]') + ',')
        self.put(self.dimrep('values_j[{0}]'))
        self.dedent(2)
        self.put(')')
        self.dedent()

    def generate_init(self):
        self.dedent()
        for line in """
    def __init__(self, kvs, geo):
        assert geo.dim == {dim}, "Geometry has wrong dimension"
        self.base_init(kvs)

        gauss = [make_iterated_quadrature(np.unique(kv.kv), self.nqp) for kv in kvs]
        gaussgrid = [g[0] for g in gauss]
        gaussweights = [g[1] for g in gauss]

        colloc = [bspline.collocation_derivs(kvs[k], gaussgrid[k], derivs={maxderiv}) for k in range({dim})]
        for k in range({dim}):
            colloc[k] = tuple(X.T.A for X in colloc[k])
        self.C = [np.stack(Cs, axis=-1) for Cs in colloc]""".splitlines():
            self.putf(line)
        self.put('')
        self.indent(2)

        if self.need_jacinv or self.need_det: self.need_jac = True
        if self.need_jac:
            self.put('geo_jac = geo.grid_jacobian(gaussgrid)')
        if self.need_det and self.need_jacinv:
            self.put('geo_det, geo_jacinv = det_and_inv(geo_jac)')
        elif self.need_det:
            self.put('geo_det = determinants(geo_jac)')
        elif self.need_jacinv:
            self.put('geo_jacinv = inverses(geo_jac)')

        self.initialize_fields()
        self.put('')
        self.dedent()

    def generate(self):
        self.putf('cdef class {classname}{dim}D(BaseAssembler{dim}D):',
                classname=self.classname)
        self.indent()
        self.put('cdef vector[double[:, :, ::1]] C       # 1D basis values. Indices: basis function, mesh point, derivative')
        # declare field variables
        for name, var in self.vars.items():
            self.putf('cdef double[{X}:1] {name}',
                    X=', '.join((self.dim + var['numdims']) * ':'),
                    name=name)
        self.generate_init()
        self.generate_kernel()
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
cpdef object generic_assemble_core_{{DIM}}d(BaseAssembler{{DIM}}D asm, bidx, bint symmetric=False):
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
cpdef object generic_assemble_core_vec_{{DIM}}d(BaseVectorAssembler{{DIM}}D asm, bidx, bint symmetric=False):
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

macros = r"""
{%- macro init_basis_vals(numderivs) -%}
        assert geo.dim == {{DIM}}, "Geometry has wrong dimension"
        self.base_init(kvs)

        gauss = [make_iterated_quadrature(np.unique(kv.kv), self.nqp) for kv in kvs]
        gaussgrid = [g[0] for g in gauss]
        gaussweights = [g[1] for g in gauss]

        colloc = [bspline.collocation_derivs(kvs[k], gaussgrid[k], derivs={{numderivs}}) for k in range({{DIM}})]
        for k in range({{DIM}}):
            colloc[k] = tuple(X.T.A for X in colloc[k])
        self.C = [np.stack(Cs, axis=-1) for Cs in colloc]
{%- endmacro %}

{%- macro assemble_impl_header(vec=0) -%}
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cdef {{ 'void' if vec else 'double' }} assemble_impl(self, size_t[{{DIM}}] i, size_t[{{DIM}}] j{% if vec %}, double result[]{% endif %}) nogil:
        cdef int k
        cdef IntInterval intv
        cdef size_t g_sta[{{DIM}}]
        cdef size_t g_end[{{DIM}}]
        cdef (double*) values_i[{{DIM}}]
        cdef (double*) values_j[{{DIM}}]

        for k in range({{DIM}}):
            intv = intersect_intervals(make_intv(self.meshsupp[k][i[k],0], self.meshsupp[k][i[k],1]),
                                       make_intv(self.meshsupp[k][j[k],0], self.meshsupp[k][j[k],1]))
            if intv.a >= intv.b:
{%- if vec -%}
{%- for k in range(vec) %}
                result[{{k}}] = 0.0
{%- endfor %}
                return          # no intersection of support
{%- else %}
                return 0.0      # no intersection of support
{%- endif %}
            g_sta[k] = self.nqp * intv.a    # start of Gauss nodes
            g_end[k] = self.nqp * intv.b    # end of Gauss nodes

            values_i[k] = &self.C[k][ i[k], g_sta[k], 0 ]
            values_j[k] = &self.C[k][ j[k], g_sta[k], 0 ]
{% endmacro %}

{%- macro make_grad(output, var, idx) -%}
{% for k in range(DIM) %}
{{output}}[{{k}}] = {{ grad_comp(var, idx, k) }}
{%- endfor %}
{% endmacro %}
"""

tmpl_divdiv_asm = Template(macros + r"""
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
cdef void combine_divdiv_{{DIM}}d(
        double[{{ dimrepeat(':') }}:1] J,
        double[{{ dimrepeat(':') }}, :, ::1] B,
        {{ dimrepeat('double* VDu{}') }},
        {{ dimrepeat('double* VDv{}') }},
        double result[]
    ) nogil:
{%- for k in range(DIM) %}
    cdef size_t n{{k}} = B.shape[{{k}}]
{%- endfor %}
    cdef size_t {{ dimrepeat('i{}') }}
    cdef double gu[{{DIM}}]
    cdef double Bgu[{{DIM}}]
    cdef double gv[{{DIM}}]
    cdef double Bgv[{{DIM}}]
    cdef double w
{% for k in range(DIM) %}
    {{ indent(k) }}for i{{k}} in range(n{{k}}):
{%- endfor %}
{%- set I = dimrepeat('i{}') %}
{{ make_grad('gu', 'VDu', 'i') | indent(4*(DIM + 1)) }}
{{ make_grad('gv', 'VDv', 'i') | indent(4*(DIM + 1)) }}

{{ matvec('Bgu', 'B[' + I + ', {i}, {j}]', 'gu') | indent(4*(DIM + 1), True) }}

{{ matvec('Bgv', 'B[' + I + ', {i}, {j}]', 'gv') | indent(4*(DIM + 1), True) }}

{{ indent(DIM) }}    w = J[{{I}}]
{% for M in range(DIM*DIM) %}
{{ indent(DIM) }}    result[{{M}}] += w * Bgv[{{ M % DIM }}] * Bgu[{{ M // DIM }}]
{%- endfor %}

cdef class DivDivAssembler{{DIM}}D(BaseVectorAssembler{{DIM}}D):
    cdef vector[double[:, :, ::1]] C            # 1D basis values. Indices: basis function, mesh point, derivative
    cdef double[{{ dimrepeat(':') }}:1] J       # weights
    cdef double[{{ dimrepeat(':') }}, :, ::1] B   # transformation matrix. Indices: DIM x mesh point, i, j

    def __init__(self, kvs, geo):
        {{ init_basis_vals(numderivs=1) }}

        geo_jac = geo.grid_jacobian(gaussgrid)
        geo_det, geo_jacinv = det_and_inv(geo_jac)
        self.J = {{ tensorprod('gaussweights') }} * np.abs(geo_det)
        self.B = np.asarray(np.swapaxes(geo_jacinv, -2, -1), order='C')

    {{ assemble_impl_header(vec=DIM*DIM) }}
        combine_divdiv_{{DIM}}d(
                self.J [ {{ dimrepeat("g_sta[{0}]:g_end[{0}]") }} ],
                self.B [ {{ dimrepeat("g_sta[{0}]:g_end[{0}]") }} ],
                {{ dimrepeat("values_i[{}]") }},
                {{ dimrepeat("values_j[{}]") }},
                result
        )

""")

class MassAsmGen(AsmGenerator):
    def __init__(self, code, dim):
        AsmGenerator.__init__(self, 'MassAssembler', code, dim, numderiv=0)
        self.register_scalar_field('J')
        self.need_val = True
        self.need_det = True

    def initialize_fields(self):
        self.putf('self.J = {gweights} * np.abs(geo_det)', gweights=self.tensorprod('gaussweights'))

    def generate_biform(self):
        self.put('result += J * v * u')


class StiffnessAsmGen(AsmGenerator):
    def __init__(self, code, dim):
        AsmGenerator.__init__(self, 'StiffnessAssembler', code, dim, numderiv=1)
        self.register_matrix_field('B', symmetric=True)
        self.need_grad = True
        self.need_det = self.need_jacinv = True

    def initialize_fields(self):
        self.putf('weights = {gweights} * np.abs(geo_det)', gweights=self.tensorprod('gaussweights'))
        self.putf('self.B = matmatT_{dim}x{dim}(geo_jacinv) * weights[{slices}, None, None]',
            slices=self.dimrep(':'))

    def generate_biform(self):
        self.add_matvecvec('B', 'gv', 'gu')


def generate(dim):
    DIM = dim

    def dimrepeat(s, sep=', ', upper=DIM):
        return sep.join([s.format(k) for k in range(upper)])

    def to_seq(i, n):
        s = i[0]
        for k in range(1, len(i)):
            s = '({0}) * {1} + {2}'.format(s, n[k], i[k])
        return s

    def extend_dim(i):
        # ex.: i = 1  ->  'None,:,None'
        slices = DIM * ['None']
        slices[i] = ':'
        return ','.join(slices)

    def tensorprod(var):
        return ' * '.join(['{0}[{1}][{2}]'.format(var, k, extend_dim(k)) for k in range(DIM)])

    def indent(num):
        return num * '    ';

    def grad_comp(var, idx, j):
        factors = [
                "{var}{k}[2*{idx}{k}+{ofs}]".format(
                    var = var,
                    idx = idx,
                    k   = k,
                    ofs = (1 if k + j + 1 == DIM else 0)
                )
                for k in range(DIM)]
        return ' * '.join(factors)

    def matvec(result, Aij, x):
        aijxj = Aij + ' * ' + x + '[{j}]'   # template for single product A[i,j]*x[j]
        components = [
                ' + '.join([
                    aijxj.format(i=i, j=j)
                    for j in range(DIM) ])
                for i in range(DIM) ]
        out = [
                result + ('[%d] = ' % i) + comp
                for i, comp in enumerate(components)
              ]
        return "\n".join(out)

    # helper variables for to_seq
    indices = ['ii[%d]' % k for k in range(DIM)]
    ndofs   = ['self.ndofs[%d]' % k for k in range(DIM)]

    code = PyCode()
    MassAsmGen(code, DIM).generate()
    StiffnessAsmGen(code, DIM).generate()

    env = locals()
    s = tmpl_generic.render(env)
    s += code.result()
    s += tmpl_divdiv_asm.render(env)
    return s

if __name__ == '__main__':
    path = os.path.join(os.path.dirname(__file__), "..", "pyiga", "assemblers.pxi")
    with open(path, 'w') as f:
        f.write('# file generated by generate-assemblers.py\n')
        f.write(generate(dim=2))
        f.write(generate(dim=3))

