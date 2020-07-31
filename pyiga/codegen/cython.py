from jinja2 import Template
from pyiga import vform

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
    def __init__(self, vform, classname, code, on_demand=False):
        self.vform = vform
        self.classname = classname
        self.code = code
        self.on_demand = on_demand
        self.dim = self.vform.dim
        self.vec = self.vform.vec
        self.updatable = tuple(inp for inp in vform.inputs if inp.updatable)

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

    def dimrep(self, s, sep=', ', dims=None):
        if dims is None:
            dims = range(self.dim)
        return sep.join([s.format(k, **self.env) for k in dims])

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

    def declare_vec(self, name, size):
        self.putf('cdef double {name}[{size}]', name=name, size=size)

    def gen_assign(self, var, expr):
        """Assign the result of an expression into a local variable; may be scalar,
        vector- or matrix-valued.
        """
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

    def field_type(self, var, contiguous=True):
        dim = self.dim if (var.depend_dims is None) else len(var.depend_dims)
        s = 'double[{X}:1]' if contiguous else 'double[{X}]'
        return s.format(X=', '.join((dim + len(var.shape)) * ':'))

    def declare_var(self, var, ref=False):
        if ref:
            if var.is_scalar():
                self.declare_scalar(var.name)
            elif var.is_vector() or var.is_matrix():
                self.declare_pointer(var.name)
        else:   # no ref - declare local storage
            if var.is_vector():
                self.declare_vec(var.name, var.shape[0])
            elif var.is_matrix():
                self.declare_vec(var.name, var.shape[0]*var.shape[1])
            else:
                self.declare_scalar(var.name)

    def declare_params(self, params, contiguous=True):
        for var in params:
            self.putf('{type} _{name},', type=self.field_type(var, contiguous=contiguous), name=var.name)

    def declare_array_vars(self, vars):
        for var in vars:
            self.putf('cdef {type} {name}', type=self.field_type(var), name=var.name)

    def load_field_var(self, var, idx, ref_only=False):
        """Load the current entry of a field var into the corresponding local variable."""
        if var.is_scalar():
            if not ref_only:
                self.putf('{name} = {entry}', name=var.name,
                    entry=self.field_var_entry(var, idx))
        elif var.is_vector() or var.is_matrix():
            self.putf('{name} = &{entry}', name=var.name,
                entry=self.field_var_entry(var, idx, first_entry=True))

    def start_loop_with_fields(self, fields_in, fields_out=[], local_vars=[]):
        fields = fields_in + fields_out

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
        for var in fields_in:
            self.load_field_var(var, 'i')
        for var in fields_out:
            # these have no values yet, only get a reference
            self.load_field_var(var, 'i', ref_only=True)
        self.put('')

        # generate code for computing local variables
        for var in local_vars:
            self.gen_assign(var, var.expr)

    def generate_kernel(self):
        # function definition
        self.cython_pragmas()
        self.put('@staticmethod')
        self.put('cdef void combine(')
        self.indent(2)
        # dimension arguments
        self.put(self.dimrep('size_t n{}') + ',')

        array_params = [var for var in self.vform.kernel_deps if var.is_array]
        local_vars   = [var for var in self.vform.kernel_deps if not var.is_array]

        # parameters
        # We will usually get slices of arrays which are therefore not contiguous.
        self.declare_params(array_params, contiguous=False)

        # arrays for basis function values/derivatives
        for bfun in self.vform.basis_funs:
            self.put(self.dimrep('double* VD%s{}' % bfun.name) + ',')

        self.put('double result[]')     # output argument
        self.dedent()
        self.put(') nogil:')

        # local variables
        if not self.vec:
            self.declare_scalar('r', '0.0')
        else:
            self.putf('cdef double* r = [ {init} ]', vec=self.vec,
                    init=', '.join(self.vec * ('0.0',)))
        self.declare_custom_variables()

        self.put('')

        ############################################################
        # main loop over all Gauss points
        self.start_loop_with_fields(array_params, local_vars=local_vars)

        # if needed, generate custom code for the bilinear form a(u,v)
        self.generate_biform_custom()

        # generate code for all expressions in the bilinear form
        if self.vec:
            for expr in self.vform.exprs:
                for i, e_i in enumerate(expr):
                    self.put(('r[%d] += ' % i) + e_i.gencode())
        else:
            for expr in self.vform.exprs:
                self.put('r += ' + expr.gencode())

        # end main loop
        for _ in range(self.dim):
            self.code.end_loop()
        ############################################################

        if not self.vec:
            self.put('result[0] = r')
        else:
            for i in range(self.vec):
                self.putf('result[{i}] = r[{i}]', i=i)

        self.end_function()

    def gen_entry_impl_header(self):
        self.cython_pragmas()
        self.putf('cdef void entry_impl(self, size_t[{dim}] i, size_t[{dim}] j, double result[]) nogil:')
        self.indent()
        self.putf('cdef int k')
        self.putf('cdef IntInterval intv')
        self.putf('cdef size_t g_sta[{dim}]')
        self.putf('cdef size_t g_end[{dim}]')
        for bfun in self.vform.basis_funs:
            self.putf('cdef (double*) values_{name}[{dim}]', name=bfun.name)

        if self.vform.arity == 1:
            idx_bfun = [('i', self.vform.basis_funs[0])]
        elif self.vform.arity == 2:
            idx_bfun = list(zip(('j', 'i'), self.vform.basis_funs))
        else:
            assert False, 'invalid arity: %d' % self.vform.arity

        for k in range(self.dim):
            if len(idx_bfun) == 1:
                idx, bfun = idx_bfun[0]
                self.putf('intv = make_intv(self.S{space}_meshsupp{k}[{idx}[{k}],0], self.S{space}_meshsupp{k}[{idx}[{k}],1])',
                        k=k, space=bfun.space, idx=idx)
            elif len(idx_bfun) == 2:
                self.putf('intv = intersect_intervals(')
                self.indent(2)
                for idx,bfun in idx_bfun:
                    self.putf('make_intv(self.S{space}_meshsupp{k}[{idx}[{k}],0], self.S{space}_meshsupp{k}[{idx}[{k}],1]),',
                            k=k, space=bfun.space, idx=idx)
                self.dedent(2)
                self.put(')')
                self.put('if intv.a >= intv.b: return   # no intersection of support')

            if self.on_demand:
                self.putf('g_sta[{k}] = self.nqp * intv.a - self.bbox_ofs[{k}]    # start of Gauss nodes', k=k)
                self.putf('g_end[{k}] = self.nqp * intv.b - self.bbox_ofs[{k}]    # end of Gauss nodes', k=k)
            else:
                self.putf('g_sta[{k}] = self.nqp * intv.a    # start of Gauss nodes', k=k)
                self.putf('g_end[{k}] = self.nqp * intv.b    # end of Gauss nodes', k=k)

            # a_ij = a(phi_j, phi_i)  -- second index (j) corresponds to first (trial) function
            for idx,bfun in idx_bfun:
                self.putf('values_{name}[{k}] = &self.S{space}_C{k}[ {idx}[{k}], g_sta[{k}], 0 ]',
                        k=k, name=bfun.name, space=bfun.space, idx=idx)
        self.put('')


    def generate_entry_impl(self):
        self.gen_entry_impl_header()

        # generate call to assembler kernel
        self.putf('{classname}.combine(', classname=self.classname)
        self.indent(2)

        # generate dimension arguments
        self.put(self.dimrep('g_end[{0}]-g_sta[{0}]') + ',')

        # generate array variable arguments
        for var in self.vform.kernel_deps:
            if var.is_array:
                self.putf('self.{name} [ {idx} ],', name=var.name,
                        idx=self.dimrep('g_sta[{0}]:g_end[{0}]', dims=var.depend_dims))

        # generate basis function value arguments
        for bfun in self.vform.basis_funs:
            self.put(self.dimrep('values_%s[{0}]' % bfun.name) + ',')

        # generate output argument
        self.put('result')

        self.dedent(2)
        self.put(')')

        self.end_function()

    def parse_src(self, var):
        s = var.src
        if isinstance(s, vform.InputField):
            if var.deriv == 0:
                if s.physical:
                    return 'np.ascontiguousarray(grid_eval_transformed(%s, self.gaussgrid, self._geo))' % s.name
                else:
                    return 'np.ascontiguousarray(grid_eval(%s, self.gaussgrid))' % s.name
            elif var.deriv == 1:
                assert not s.physical, 'Jacobian of physical input field not implemented'
                return 'np.ascontiguousarray(%s.grid_jacobian(self.gaussgrid))' % s.name
            elif var.deriv == 2:
                assert not s.physical, 'Hessian of physical input field not implemented'
                return 'np.ascontiguousarray(%s.grid_hessian(self.gaussgrid))' % s.name
            else:
                assert False, 'invalid derivative %s for input field %s' % (var.deriv, s.name)
        elif s.startswith('@gaussweights'):
            return s[1:]
        else:
            assert False, 'invalid source %s for var %s' % (s, var.name)

    def generate_metadata(self):
        # generate an 'inputs' property which returns a name->shape dict
        self.put('@classmethod')
        self.put('def inputs(cls):')
        self.indent()
        self.put('return {')
        for inp in self.vform.inputs:
            self.putf("    '{name}': {shp},", name=inp.name, shp=inp.shape)
        self.put('}')
        self.dedent()
        self.put('')

    def generate_init(self):
        vf = self.vform

        used_spaces = sorted(set(bf.space for bf in vf.basis_funs))
        assert len(used_spaces) in (1,2), 'Number of spaces should be 1 or 2'
        used_kvs = tuple('kvs%d' % sp for sp in used_spaces)
        input_args = ', '.join(inp.name for inp in vf.inputs)

        self.putf('def __init__(self, {kvs}, {inp}{bbox}):', kvs=', '.join(used_kvs),
                bbox = ', bbox' if self.on_demand else '',
                inp=input_args)
        self.indent()

        self.putf('self.arity = {ar}', ar=vf.arity)
        self.putf('self.nqp = max([kv.p for kv in {kvs}]) + 1', kvs=' + '.join(used_kvs))
        if len(used_spaces) == 1:
            self.put('kvs1 = kvs0')
        self.put('self.kvs = (kvs0, kvs1)')

        if self.vec:
            numcomp = vf.num_components()
            numcomp += (2 - len(numcomp)) * (0,) # pad to 2
            numcomp = '(' + ', '.join(str(nc) for nc in numcomp) + ',)'
            self.put("self.numcomp[:] = " + numcomp)

        self.putf('assert geo.sdim == {dim}, "Geometry has wrong source dimension"')
        self.putf('assert geo.dim == {geo_dim}, "Geometry has wrong dimension"')
        self.put('self._geo = geo')
        self.put('')
        self.put('# NB: we assume all kvs result in the same mesh')

        if self.on_demand:
            # NB: bb[1] is the exclusive upper cell index; i.e., the last cell is bb[1]-1
            # kv.mesh[k] is the starting point of cell k or the end point of cell k-1
            # kv.mesh[bb[1]] is the end point of cell bb[1]-1;
            # we need to INCLUDE that end point!
            self.put('gaussgrid, gaussweights = make_tensor_quadrature([kv.mesh[bb[0]:bb[1]+1] for (kv,bb) in zip(kvs0,bbox)], self.nqp)')
        else:
            self.put('gaussgrid, gaussweights = make_tensor_quadrature([kv.mesh for kv in kvs0], self.nqp)')

        self.put('self.gaussgrid = gaussgrid')
        self.put('N = tuple(gg.shape[0] for gg in gaussgrid)  # grid dimensions')

        if self.on_demand:
            self.put('self.bbox_ofs[:] = tuple(bb[0] * self.nqp for bb in bbox)')

        self.put('')

        for sp in (0,1):        # TODO: do this only for used_spaces? crashes currently
            kvs = 'kvs%d' % sp
            self.putf('assert len({kvs}) == {dim}, "Assembler requires {dim} knot vectors"', kvs=kvs)
            self.putf('self.S{sp}_ndofs[:] = [kv.numdofs for kv in {kvs}]', sp=sp, kvs=kvs)
            for k in range(self.dim):
                self.putf('self.S{sp}_meshsupp{k} = {kvs}[{k}].mesh_support_idx_all()', sp=sp, k=k, kvs=kvs)
                self.putf('self.S{sp}_C{k} = compute_values_derivs(kvs{sp}[{k}], gaussgrid[{k}], derivs={maxderiv})',
                        k=k, sp=sp)
        self.put('')

        # declare array storage for non-global variables
        self.declare_array_vars(var for var in self.vform.precomp_deps
                if var.is_array and not var.is_global)

        def array_var_ref(var):
            assert var.is_array
            return ('self.' if var.is_global else '') + var.name

        # declare/initialize array variables
        for var in vf.linear_deps:
            # exclude basis function nodes
            if not isinstance(var, vform.BasisFun) and var.is_array:
                arr = array_var_ref(var)
                if var.src:
                    self.putf("{arr} = {src}", arr=arr, src=self.parse_src(var))
                elif var.expr:  # custom precomputed field var
                    self.putf("{arr} = np.empty(N + {shape})", arr=arr, shape=var.shape)

        if vf.precomp:
            # call precompute function
            self.putf('{classname}.precompute_fields(', classname=self.classname)
            self.indent(2)
            # generate size arguments
            self.put(self.dimrep('gaussgrid[{}].shape[0]') + ',')
            # generate arguments for input and output fields
            for var in vf.precomp_deps + vf.precomp:
                self.put(array_var_ref(var) + ',')
            self.dedent(2)
            self.put(')')

        self.initialize_custom_fields()
        self.end_function()

    def field_var_entry(self, var, idx, first_entry=False):
        """Generate a reference to the current entry in the given field variable."""
        I = self.dimrep(idx + '{}', dims=var.depend_dims)
        if first_entry:
            I += (len(var.shape) * ', 0')
        return '_%s[%s]' % (var.name, I)

    def generate_precomp(self):
        vf = self.vform

        # function header
        self.cython_pragmas()
        self.put('@staticmethod')
        self.put('cdef void precompute_fields(')
        self.indent(2)
        self.put('# dimensions')
        self.put(self.dimrep('size_t n{}') + ',')
        self.put('# input')
        self.declare_params(vf.precomp_deps)
        self.put('# output')
        self.declare_params(vf.precomp)
        self.dedent()
        self.put(') nogil:')

        # start main loop
        self.start_loop_with_fields(vf.precomp_deps, fields_out=vf.precomp, local_vars=vf.precomp_locals)

        # generate assignment statements
        for var in vf.precomp:
            self.gen_assign(var, var.expr)
            if var.is_scalar():
                # for scalars, we need to explicitly copy the computed value into
                # the field array; vectors and matrices use pointers directly
                self.putf('{entry} = {name}', entry=self.field_var_entry(var, 'i'), name=var.name)

        # end main loop
        for _ in range(self.dim):
            self.code.end_loop()
        self.end_function()

    def generate_update(self):
        vf = self.vform

        self.putf('def update(self, {args}):',
                args=', '.join('%s=None' % inp.name for inp in self.updatable))
        self.indent()

        # declare/initialize array variables
        for var in vf.linear_deps:
            if not isinstance(var, vform.BasisFun) and var.src in self.updatable:
                assert var.is_array and var.is_global, 'only global array vars can be updated'
                self.putf("if {name}:", name=var.src.name)
                self.indent()
                self.putf("{arr} = {src}", arr='self.'+var.name, src=self.parse_src(var))
                self.dedent()
        self.end_function()

    # main code generation entry point

    def generate(self):
        self.vform.finalize(do_precompute=not self.on_demand)
        self.numderiv = self.vform.find_max_deriv()

        self.env = {
            'dim': self.vform.dim,
            'geo_dim': self.vform.geo_dim,
            'maxderiv': self.numderiv,
        }

        baseclass = 'BaseVectorAssembler' if self.vec else 'BaseAssembler'
        self.putf('cdef class {classname}({base}{dim}D):',
                classname=self.classname, base=baseclass)
        self.indent()

        # declare array storage for global variables
        self.declare_array_vars(var for var in self.vform.kernel_deps
                if var.is_array and var.is_global)
        self.put('')

        # generate metadata accessors
        self.generate_metadata()

        # generate methods
        self.generate_init()
        self.put('')
        if self.vform.precomp:
            self.generate_precomp()
            self.put('')
        self.generate_kernel()
        self.put('')
        self.generate_entry_impl()

        if self.updatable:
            self.put('')
            self.generate_update()

        # end of class definition
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

cdef class BaseAssembler{{DIM}}D:
    cdef readonly int arity
    cdef int nqp
    {%- for S in range(2) %}
    cdef size_t[{{DIM}}] S{{S}}_ndofs
    {%- for k in range(DIM) %}
    cdef ssize_t[:,::1] S{{S}}_meshsupp{{k}}
    cdef double[:, :, ::1] S{{S}}_C{{k}}
    {%- endfor %}
    {%- endfor %}
    cdef readonly tuple kvs
    cdef object _geo
    cdef tuple gaussgrid
    cdef size_t[{{DIM}}] bbox_ofs

    cdef void entry_impl(self, size_t[{{DIM}}] i, size_t[{{DIM}}] j, double result[]) nogil:
        pass

    cpdef double entry1(self, size_t i):
        """Compute an entry of the vector to be assembled."""
        if self.arity != 1:
            return 0.0
        cdef size_t[{{DIM}}] I
        cdef double result = 0.0
        with nogil:
            from_seq{{DIM}}(i, self.S0_ndofs, I)
            self.entry_impl(I, <size_t*>0, &result)
            return result

    cpdef double entry(self, size_t i, size_t j):
        """Compute an entry of the matrix."""
        if self.arity != 2:
            return 0.0
        cdef size_t[{{DIM}}] I, J
        cdef double result = 0.0
        with nogil:
            from_seq{{DIM}}(i, self.S1_ndofs, I)
            from_seq{{DIM}}(j, self.S0_ndofs, J)
            self.entry_impl(I, J, &result)
            return result

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void multi_entries_chunk(self, size_t[:,::1] idx_arr, double[::1] out) nogil:
        if self.arity != 2:
            return
        cdef size_t[{{DIM}}] I, J
        cdef size_t k

        for k in range(idx_arr.shape[0]):
            from_seq{{DIM}}(idx_arr[k,0], self.S1_ndofs, I)
            from_seq{{DIM}}(idx_arr[k,1], self.S0_ndofs, J)
            self.entry_impl(I, J, &out[k])

    def multi_entries1(self, indices):
        """Compute all entries given by `indices`.

        Args:
            indices: a sequence or `ndarray` of vector indices to compute
        """
        if self.arity != 1:
            return None
        cdef size_t[::1] idx_arr
        if isinstance(indices, np.ndarray):
            idx_arr = np.asarray(indices, order='C', dtype=np.uintp)
        else:   # possibly given as iterator
            idx_arr = np.array(list(indices), dtype=np.uintp)

        cdef double[::1] result = np.empty(idx_arr.shape[0])
        cdef int i
        for i in range(idx_arr.shape[0]):
            result[i] = self.entry1(idx_arr[i])
        return result

    def multi_entries(self, indices):
        """Compute all entries given by `indices`.

        Args:
            indices: a sequence of `(i,j)` pairs or an `ndarray`
            of size `N x 2`.
        """
        if self.arity == 1:
            return self.multi_entries1(indices)
        elif self.arity != 2:
            return None
        cdef size_t[:,::1] idx_arr
        if isinstance(indices, np.ndarray):
            idx_arr = np.asarray(indices, order='C', dtype=np.uintp)
        else:   # possibly given as iterator
            idx_arr = np.array(list(indices), dtype=np.uintp)

        _result = np.zeros(idx_arr.shape[0])
        cdef double[::1] result = _result

        num_threads = pyiga.get_max_threads()
        if num_threads <= 1:
            self.multi_entries_chunk(idx_arr, result)
        else:
            thread_pool = get_thread_pool()

            def asm_chunk(idxchunk, out):
                cdef size_t[:, ::1] idxchunk_ = idxchunk
                cdef double[::1] out_ = out
                with nogil:
                    self.multi_entries_chunk(idxchunk_, out_)

            results = thread_pool.map(asm_chunk,
                        chunk_tasks(idx_arr, num_threads),
                        chunk_tasks(result, num_threads))
            list(results)   # wait for threads to finish
        return _result

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def assemble_vector(self):
        if self.arity != 1:
            return None
        result = np.zeros(tuple(self.S0_ndofs), order='C')
        cdef double[{{ dimrepeat(':') }}:1] _result = result
        cdef double* out = &_result[ {{ dimrepeat('0') }} ]

        cdef size_t[{{DIM}}] I, zero
        {{ dimrepeat('zero[{}]', sep=' = ') }} = 0
        {{ dimrepeat('I[{}]', sep=' = ') }} = 0
        with nogil:
            while True:
               self.entry_impl(I, <size_t*>0, out)
               out += 1
               if not next_lexicographic{{DIM}}(I, zero, self.S0_ndofs):
                   break
        return result

    def entry_func_ptr(self):
        return pycapsule.PyCapsule_New(<void*>_entry_func_{{DIM}}d, "entryfunc", NULL)


# helper function for fast low-rank assembler
cdef double _entry_func_{{DIM}}d(size_t i, size_t j, void * data):
    return (<BaseAssembler{{DIM}}D>data).entry(i, j)



cdef class BaseVectorAssembler{{DIM}}D:
    cdef readonly int arity
    cdef int nqp
    {%- for S in range(2) %}
    cdef size_t[{{DIM}}] S{{S}}_ndofs
    {%- for k in range(DIM) %}
    cdef ssize_t[:,::1] S{{S}}_meshsupp{{k}}
    cdef double[:, :, ::1] S{{S}}_C{{k}}
    {%- endfor %}
    {%- endfor %}
    cdef size_t[2] numcomp  # number of vector components for trial and test functions
    cdef readonly tuple kvs
    cdef object _geo
    cdef tuple gaussgrid

    def num_components(self):
        return self.numcomp[0], self.numcomp[1]

    cdef void entry_impl(self, size_t[{{DIM}}] i, size_t[{{DIM}}] j, double result[]) nogil:
        pass

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void multi_blocks_chunk(self, size_t[:,::1] idx_arr, double[:,:,::1] out) nogil:
        if self.arity != 2:
            return
        cdef size_t[{{DIM}}] I, J
        cdef size_t k

        for k in range(idx_arr.shape[0]):
            from_seq{{DIM}}(idx_arr[k,0], self.S1_ndofs, I)
            from_seq{{DIM}}(idx_arr[k,1], self.S0_ndofs, J)
            self.entry_impl(I, J, &out[k, 0, 0])

    def multi_blocks(self, indices):
        """Compute all blocks with the given `indices`.

        Args:
            indices: a sequence of `(i,j)` pairs or an `ndarray`
            of size `N x 2`.
        Returns:
            an array of size `N x numcomp[0] x numcomp[1]`
        """
        if self.arity == 1:
            return None #self.multi_entries1(indices)
        elif self.arity != 2:
            return None
        cdef size_t[:,::1] idx_arr
        if isinstance(indices, np.ndarray):
            idx_arr = np.asarray(indices, order='C', dtype=np.uintp)
        else:   # possibly given as iterator
            idx_arr = np.array(list(indices), dtype=np.uintp)

        _result = np.zeros((idx_arr.shape[0], self.numcomp[0], self.numcomp[1]))
        cdef double[:, :, ::1] result = _result

        num_threads = pyiga.get_max_threads()
        if num_threads <= 1:
            self.multi_blocks_chunk(idx_arr, result)
        else:
            thread_pool = get_thread_pool()

            def asm_chunk(idxchunk, out):
                cdef size_t[:, ::1] idxchunk_ = idxchunk
                cdef double[:, :, ::1] out_ = out
                with nogil:
                    self.multi_blocks_chunk(idxchunk_, out_)

            results = thread_pool.map(asm_chunk,
                        chunk_tasks(idx_arr, num_threads),
                        chunk_tasks(result, num_threads))
            list(results)   # wait for threads to finish
        return _result

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def assemble_vector(self):
        if self.arity != 1:
            return None
        result = np.zeros(tuple(self.S0_ndofs) + (self.numcomp[0],), order='C')
        cdef double[{{ dimrepeat(':') }}, ::1] _result = result
        cdef double* out = &_result[ {{ dimrepeat('0') }}, 0 ]

        cdef size_t[{{DIM}}] I, zero
        {{ dimrepeat('zero[{}]', sep=' = ') }} = 0
        {{ dimrepeat('I[{}]', sep=' = ') }} = 0
        with nogil:
            while True:
               self.entry_impl(I, <size_t*>0, out)
               out += self.numcomp[0]
               if not next_lexicographic{{DIM}}(I, zero, self.S0_ndofs):
                   break
        return result


@cython.boundscheck(False)
@cython.wraparound(False)
def generic_assemble_core_vec_{{DIM}}d(BaseVectorAssembler{{DIM}}D asm, bidx, bint symmetric=False):
    if asm.arity != 2:
        return None
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
{{ indent(DIM) }}asm.entry_impl(i, j, &entries[ {{ dimrepeat('mu{}') }}, 0 ])

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

    def indent(num):
        return num * '    ';

    return tmpl_generic.render(locals())

def preamble():
    return \
"""# cython: profile=False
# cython: linetrace=False
# cython: binding=False

#######################
# Autogenerated code. #
# Do not modify.      #
#######################

cimport cython
from libc.math cimport fabs, sqrt, exp, log, sin, cos, tan

import numpy as np
cimport numpy as np

from pyiga.quadrature import make_tensor_quadrature

from pyiga.assemble_tools_cy cimport (
    BaseAssembler2D, BaseAssembler3D,
    BaseVectorAssembler2D, BaseVectorAssembler3D,
    IntInterval, make_intv, intersect_intervals,
    next_lexicographic2, next_lexicographic3,
)
from pyiga.assemble_tools import compute_values_derivs
from pyiga.utils import LazyCachingArray, grid_eval, grid_eval_transformed

"""
