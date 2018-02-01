from jinja2 import Template

class CodeGen:
    """Basic code generation helper for Cython."""
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
    """Generates a Cython assembler class from an abstract :class:`pyiga.vform.VForm`."""
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

    def declare_params(self, params):
        for var in params:
            self.putf('{type} _{name},', type=self.field_type(var), name=var.name)

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
        self.declare_params(field_params)

        # arrays for basis function values/derivatives
        for bfun in self.vform.basis_funs:
            self.put(self.dimrep('double* VD%s{}' % bfun.name) + ',')

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
        self.putf(funcdecl)
        self.indent()
        self.putf('cdef int k')
        self.putf('cdef IntInterval intv')
        self.putf('cdef size_t g_sta[{dim}]')
        self.putf('cdef size_t g_end[{dim}]')
        self.putf('cdef (double*) values_i[{dim}]')
        self.putf('cdef (double*) values_j[{dim}]')

        for k in range(self.dim):
            self.putf(
        """intv = intersect_intervals(make_intv(self.S0.meshsupp{k}[i[{k}],0], self.S0.meshsupp{k}[i[{k}],1]),
                                      make_intv(self.S0.meshsupp{k}[j[{k}],0], self.S0.meshsupp{k}[j[{k}],1]))""", k=k)
            self.put('if intv.a >= intv.b: return ' + zeroret + '  # no intersection of support')
            self.putf('g_sta[{k}] = self.nqp * intv.a    # start of Gauss nodes', k=k)
            self.putf('g_end[{k}] = self.nqp * intv.b    # end of Gauss nodes', k=k)

            self.putf('values_i[{k}] = &self.C{k}[ i[{k}], g_sta[{k}], 0 ]', k=k)
            self.putf('values_j[{k}] = &self.C{k}[ j[{k}], g_sta[{k}], 0 ]', k=k)
        self.put('')


    def generate_assemble_impl(self):
        self.gen_assemble_impl_header()

        # generate call to assembler kernel
        if self.vec:
            self.putf('{classname}.combine(', classname=self.classname)
        else:
            self.putf('return {classname}.combine(', classname=self.classname)
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

        if self.vec:
            numcomp = '(' + ', '.join(str(nc) for nc in vf.num_components()) + ',)'
            self.putf('self.base_init(kvs, {nc})', nc=numcomp)
        else:
            self.putf('self.base_init(kvs)')

        for line in \
"""assert geo.dim == {dim}, "Geometry has wrong dimension"

gaussgrid, gaussweights = make_tensor_quadrature([kv.mesh for kv in kvs], self.nqp)
N = tuple(gg.shape[0] for gg in gaussgrid)  # grid dimensions""".splitlines():
            self.putf(line)
        self.put('')

        for k in range(self.dim):
            self.putf('self.C{k} = compute_values_derivs(kvs[{k}], gaussgrid[{k}], derivs={maxderiv})', k=k)
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
            self.putf('{classname}.precompute_fields(', classname=self.classname)
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
        self.declare_params(vf.precomp_deps)
        self.put('# output')
        self.declare_params(vf.precomp)
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
        self.putf('cdef class {classname}({base}{dim}D):',
                classname=self.classname, base=baseclass)
        self.indent()

        # declare instance variables
        self.put('# 1D basis values. Indices: basis function, mesh point, derivative')
        for k in range(self.dim):
            self.putf('cdef double[:, :, ::1] C{k}', k=k)

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


################################################################################
# Generic templates for assembler infrastructure
################################################################################

tmpl_generic = Template(r'''
################################################################################
# {{DIM}}D Assemblers
################################################################################

cdef struct SpaceInfo{{DIM}}:
    size_t[{{DIM}}] ndofs
    int[{{DIM}}] p
    {%- for k in range(DIM) %}
    ssize_t[:,::1] meshsupp{{k}}
    {%- endfor %}

cdef class BaseAssembler{{DIM}}D:
    cdef int nqp
    cdef SpaceInfo{{DIM}} S0

    cdef void base_init(self, kvs):
        assert len(kvs) == {{DIM}}, "Assembler requires {{DIM}} knot vectors"
        self.nqp = max([kv.p for kv in kvs]) + 1
        self.S0.ndofs[:] = [kv.numdofs for kv in kvs]
        self.S0.p[:]     = [kv.p for kv in kvs]
        {%- for k in range(DIM) %}
        self.S0.meshsupp{{k}} = kvs[{{k}}].mesh_support_idx_all()
        {%- endfor %}

    cdef inline size_t to_seq(self, size_t[{{DIM}}] ii) nogil:
        # by convention, the order of indices is (y,x)
        return {{ to_seq(indices, ndofs) }}

    cdef double assemble_impl(self, size_t[{{DIM}}] i, size_t[{{DIM}}] j) nogil:
        return -9999.99  # Not implemented

    cpdef double assemble(self, size_t i, size_t j):
        cdef size_t[{{DIM}}] I, J
        with nogil:
            from_seq{{DIM}}(i, self.S0.ndofs, I)
            from_seq{{DIM}}(j, self.S0.ndofs, J)
            return self.assemble_impl(I, J)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void multi_assemble_chunk(self, size_t[:,::1] idx_arr, double[::1] out) nogil:
        cdef size_t[{{DIM}}] I, J
        cdef size_t k

        for k in range(idx_arr.shape[0]):
            from_seq{{DIM}}(idx_arr[k,0], self.S0.ndofs, I)
            from_seq{{DIM}}(idx_arr[k,1], self.S0.ndofs, J)
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

            def asm_chunk(idxchunk, out):
                cdef size_t[:, ::1] idxchunk_ = idxchunk
                cdef double[::1] out_ = out
                with nogil:
                    self.multi_assemble_chunk(idxchunk_, out_)

            results = thread_pool.map(asm_chunk,
                        chunk_tasks(idx_arr, num_threads),
                        chunk_tasks(result, num_threads))
            list(results)   # wait for threads to finish
        return result

    def entry_func_ptr(self):
        return pycapsule.PyCapsule_New(<void*>_entry_func_{{DIM}}d, "entryfunc", NULL)


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
        tuple(asm.S0.ndofs),
        tuple(asm.S0.p)
    )
    X = generic_assemble_core_{{DIM}}d(asm, mlb.bidx, symmetric=symmetric)
    mlb.data = X
    return mlb


# helper function for fast low-rank assembler
cdef double _entry_func_{{DIM}}d(size_t i, size_t j, void * data):
    return (<BaseAssembler{{DIM}}D>data).assemble(i, j)



cdef class BaseVectorAssembler{{DIM}}D:
    cdef int nqp
    cdef SpaceInfo{{DIM}} S0
    cdef size_t[2] numcomp  # number of vector components for trial and test functions

    cdef void base_init(self, kvs, numcomp):
        assert len(kvs) == {{DIM}}, "Assembler requires {{DIM}} knot vectors"
        self.nqp = max([kv.p for kv in kvs]) + 1
        self.S0.ndofs[:] = [kv.numdofs for kv in kvs]
        {%- for k in range(DIM) %}
        self.S0.meshsupp{{k}} = kvs[{{k}}].mesh_support_idx_all()
        {%- endfor %}
        self.numcomp[:] = numcomp
        assert self.numcomp[0] == self.numcomp[1], 'Only square matrices currently implemented'

    def num_components(self):
        return self.numcomp[0], self.numcomp[1]

    cdef inline size_t to_seq(self, size_t[{{DIM + 1}}] ii) nogil:
        return {{ to_seq(indices + ['ii[%d]' % DIM], ndofs + ['self.numcomp[0]']) }}

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef inline void from_seq(self, size_t i, size_t[{{DIM + 1}}] out) nogil:
        out[{{DIM}}] = i % self.numcomp[0]
        i /= self.numcomp[0]
        {%- for k in range(1, DIM)|reverse %}
        out[{{k}}] = i % self.S0.ndofs[{{k}}]
        i /= self.S0.ndofs[{{k}}]
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
    cdef size_t[2] numcomp

    {{ dimrepeat('bidx{}') }} = bidx
    {{ dimrepeat('MU{}') }} = {{ dimrepeat('bidx{}.shape[0]') }}

    cdef size_t[::1] {{ dimrepeat('transp{}') }}
    if symmetric:
    {%- for k in range(DIM) %}
        transp{{k}} = get_transpose_idx_for_bidx(bidx{{k}})
    {%- endfor %}
    else:
        {{ dimrepeat('transp{}', sep=' = ') }} = None

    numcomp[:] = asm.num_components()
    assert numcomp[0] == numcomp[1], 'only square matrices currently implemented'
    entries = np.zeros(({{ dimrepeat('MU{}') }}, numcomp[0]*numcomp[1]))

    cdef int num_threads = pyiga.get_max_threads()

    for mu0 in prange(MU0, num_threads=num_threads, nogil=True):
        _asm_core_vec_{{DIM}}d_kernel(asm, symmetric,
            {{ dimrepeat('bidx{}') }},
            {{ dimrepeat('transp{}') }},
            numcomp,
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
    size_t[2] numcomp,
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
{{ indent(DIM) }}        for row in range(numcomp[1]):
{{ indent(DIM) }}            for col in range(numcomp[0]):
{{ indent(DIM) }}                entries[{{ dimrepeat('transp{0}[mu{0}]') }}, col*numcomp[0] + row] = entries[{{ dimrepeat('mu{}') }}, row*numcomp[0] + col]

''')


def generate_generic(dim):
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
    ndofs   = ['self.S0.ndofs[%d]' % k for k in range(DIM)]

    return tmpl_generic.render(locals())

def preamble():
    return \
"""# cython: language_level=3
# cython: profile=False
# cython: linetrace=False
# cython: binding=False

#######################
# Autogenerated code. #
# Do not modify.      #
#######################

cimport cython
from libc.math cimport fabs

import numpy as np
cimport numpy as np

from pyiga.quadrature import make_tensor_quadrature

from pyiga.assemble_tools_cy cimport (
    BaseAssembler2D, BaseAssembler3D,
    BaseVectorAssembler2D, BaseVectorAssembler3D,
    IntInterval, make_intv, intersect_intervals,
)
from pyiga.assemble_tools_cy import compute_values_derivs

"""
