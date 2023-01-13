from jinja2 import Template
from pyiga import vform
from functools import reduce
import operator
import numpy as np

# Some notes on AsmVars:
#
# - AsmVars with a src:
#   - InputField: var is computed during init and then stored either in `fields`
#       (for kernel deps) or in `temp_fields` (only used during precompute)
#   - Parameter: var is allocated in `constants` and must be set before assembling
#       (there are no `temp_constants` since storage overhead is trivial)
#
# - AsmVars with an expr:
#   - are computed either in precompute or in kernel
#   - are allocated in `fields` or `constants` if precomputed AND a kernel dependency
#       (is_global=True)
#   - may be local vars in precompute or kernel (no global storage) if only used
#       in one of the two functions

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


class CodegenVisitor:
    def gencode(self, expr):
        """Generate code for a given expression."""
        dispatch = {
                vform.ConstExpr: self.gencode_const,
                vform.VarRefExpr: self.gencode_varref,
                vform.NegExpr: self.gencode_neg,
                vform.BuiltinFuncExpr: self.gencode_builtinfunc,
                vform.ScalarOperExpr: self.gencode_scalaroper,
                vform.PartialDerivExpr: self.gencode_partialderiv,
                vform.GaussWeightExpr: self.gencode_gaussweight,
        }
        return dispatch[type(expr)](expr)

    def gencode_const(self, expr):
        return repr(expr.value)

    def gencode_varref(self, expr):
        assert expr.is_scalar()
        return self.var_ref(expr.var, expr.I)

    def gencode_neg(self, expr):
        return '-' + self.gencode(expr.x)

    func_to_code = {
            'abs' : 'fabs',
    }

    def gencode_builtinfunc(self, expr):
        f = self.func_to_code.get(expr.funcname, expr.funcname)
        return '%s(%s)' % (f, self.gencode(expr.x))

    def gencode_scalaroper(self, expr):
        sep = ' ' + expr.oper + ' '
        return '(' + sep.join(self.gencode(x) for x in expr.children) + ')'

    def gencode_partialderiv(self, expr):
        assert not expr.physical, 'cannot generate code for physical derivative'
        # gen_pderiv is defined in AsmGenerator
        return self.gen_pderiv(expr.basisfun, expr.D)

    def gencode_gaussweight(self, expr):
        ax = expr.axis
        return '_gw{0}[i{0}]'.format(ax)

def storage_size(var):
    """Size of the flattened storage of a field variable."""
    if len(var.shape) == 2 and var.symmetric:
        m, n = var.shape
        assert m == n
        return m * (m + 1) // 2
    else:
        return reduce(operator.mul, var.shape, 1) # number of scalar entries

def storage_index(var, I):
    """Index into the flattened storage of a field variable."""
    if var.shape == ():
        assert I == ()
        return 0
    else:
        if len(var.shape) == 2 and var.symmetric:
            return vform.sym_index_to_seq(var.shape[0], *I)
        else:
            return np.ravel_multi_index(I, var.shape)

def allocate_array(entries):
    ofs = 0
    info = {}
    for entry in entries:
        sz = storage_size(entry)
        info[entry.name] = (entry, sz, ofs)
        ofs += sz
    return info, ofs


class AsmGenerator(CodegenVisitor):
    """Generates a Cython assembler class from an abstract :class:`pyiga.vform.VForm`."""
    def __init__(self, vform, classname, code, on_demand=False):
        self.vform = vform
        self.classname = classname
        self.code = code
        self.on_demand = on_demand
        self.dim = self.vform.dim
        self.vec = self.vform.vec
        self.updatable = tuple(inp for inp in vform.inputs if inp.updatable)

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

    def var_ref(self, var, I):
        if var.name in self.global_info:
            var, sz, ofs = self.global_info[var.name]
            idx = ofs + storage_index(var, I)
            return 'fields[{i}]'.format(i=idx)
        elif var.name in self.temp_info:
            var, sz, ofs = self.temp_info[var.name]
            idx = ofs + storage_index(var, I)
            return 'temp_fields[{i}]'.format(i=idx)
        elif var.name in self.constant_info:
            par, sz, ofs = self.constant_info[var.name]
            idx = ofs + storage_index(par, I)
            return 'constants[{i}]'.format(i=idx)
        else:   # it's a non-global, referred to by name
            if var.is_scalar():
                return var.name
            else:
                return '{name}[{k}]'.format(name=var.name, k=storage_index(var, I))

    def gen_assign(self, var, expr):
        """Assign the result of an expression into a local variable; may be scalar,
        vector- or matrix-valued.
        """
        self.put('# ' + var.name)
        if expr.is_scalar():
            self.put(self.var_ref(var, ()) + ' = ' + self.gencode(expr))
        elif expr.is_vector():
            for k in range(expr.shape[0]):
                lhs = self.var_ref(var, (k,))
                rhs = self.gencode(expr[k])
                self.put(lhs + ' = ' + rhs)
        elif expr.is_matrix():
            m, n = expr.shape
            for i in range(m):
                for j in range(n):
                    if var.symmetric and i > j:
                        continue
                    lhs = self.var_ref(var, (i, j))
                    rhs = self.gencode(expr[i,j])
                    self.put(lhs + ' = ' + rhs)
        else:
            assert False, 'unsupported shape'

    def gen_var_debug_dump(self, var):
        """Generate code to write the value of `var` to the screen. Requires
            ``from libc.stdio cimport printf``
        in the preamble.
        """
        def vec_fmt(n):
            return '[' + ' '.join(['%f' for k in range(n)]) + ']'

        expr = var.expr
        if expr.is_scalar():
            self.putf(r'printf("{name}: %f\n", {ref})', name=var.name, ref=self.var_ref(var, ()))
        elif expr.is_vector():
            n = expr.shape[0]
            refs = ', '.join([self.var_ref(var, (k,)) for k in range(n)])
            self.putf(r'printf("{name}: {fmt}\n", {refs})', name=var.name, fmt=vec_fmt(n), refs=refs)
        elif expr.is_matrix():
            m, n = expr.shape
            refs = ', '.join([self.var_ref(var, (i, j)) for i in range(m) for j in range(n)])
            fmt = m * vec_fmt(n)
            self.putf(r'printf("{name}: [{fmt}]\n", {refs})', name=var.name, fmt=fmt, refs=refs)
        else:
            assert False, 'unsupported shape'

    def cython_pragmas(self):
        self.put('@cython.boundscheck(False)')
        self.put('@cython.wraparound(False)')
        self.put('@cython.initializedcheck(False)')

    def field_type(self):   # array type for field variables
        return 'double[{X}, ::1]'.format(X=self.dimrep(':'))

    def load_field_var(self, var, idx):
        """Load the current entry of a field var into the corresponding local variable."""
        if var.is_scalar():
            self.putf('{name} = {entry}', name=var.name,
                    entry=self.field_var_entry(var, idx))
        elif var.is_vector() or var.is_matrix():
            self.putf('{name} = &{entry}', name=var.name,
                entry=self.field_var_entry(var, idx, first_entry=True))

    def declare_var(self, var, ref=False):
        if ref:
            if var.is_scalar():
                self.declare_scalar(var.name)
            elif var.is_vector() or var.is_matrix():
                self.declare_pointer(var.name)
        else:   # no ref - declare local storage
            if var.is_scalar():
                self.declare_scalar(var.name)
            elif var.is_vector() or var.is_matrix():
                self.declare_vec(var.name, storage_size(var))

    def start_loop_with_fields(self, local_vars=[], temp=False):
        # temp storage for local variables
        for var in local_vars:
            if var.expr and not var.is_global:  # globals already have storage
                self.declare_var(var)

        # generate assignment statements for constants
        for var in local_vars:
            if var.expr and var.scope == vform.Scope.CONSTANT:
                self.gen_assign(var, var.expr)

        self.declare_pointer('fields')
        if temp:
            self.declare_pointer('temp_fields')

        # declare iteration indices
        for k in range(self.dim):
            self.declare_index('i%d' % k)

        # start the for loop
        self.put('')
        for k in range(self.dim):
            self.code.for_loop('i%d' % k, 'n%d' % k)

        self.putf('fields = &_fields[{idx}, 0]', idx=self.dimrep('i{}'))
        if temp:
            self.putf('temp_fields = &_temp_fields[{idx}, 0]', idx=self.dimrep('i{}'))
        self.put('')

        # generate code for computing field variables
        for var in local_vars:
            if var.expr and var.scope != vform.Scope.CONSTANT:
                self.gen_assign(var, var.expr)

    def generate_kernel(self):
        # function definition
        self.cython_pragmas()
        self.put('@staticmethod')
        self.put('cdef void combine(')
        self.indent(2)
        # dimension arguments
        self.put(self.dimrep('size_t n{}') + ',')
        self.put(self.dimrep('double* _gw{}') + ',')

        # input: 'fields' contains all global arrays
        self.put('double[' + self.dimrep(':') + ', :] _fields,')
        if self.num_constants > 0:
            self.put('double constants[],')    # array for constant parameters

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
        # exclude globals, they have already been computed at this point
        local_vars = [var for var in self.vform.kernel_deps if not var.is_global]
        self.start_loop_with_fields(local_vars=local_vars)

        # if needed, generate custom code for the bilinear form a(u,v)
        self.generate_biform_custom()

        # generate code for all expressions in the bilinear form
        if self.vec:
            for expr in self.vform.exprs:
                for i, e_i in enumerate(expr):
                    self.put(('r[%d] += ' % i) + self.gencode(e_i))
        else:
            for expr in self.vform.exprs:
                self.put('r += ' + self.gencode(expr))

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
                self.putf('g_sta[{k}] = intv.a - self.bbox_ofs[{k}]    # start of Gauss nodes', k=k)
                self.putf('g_end[{k}] = intv.b - self.bbox_ofs[{k}]    # end of Gauss nodes', k=k)
            else:
                self.putf('g_sta[{k}] = intv.a    # start of Gauss nodes', k=k)
                self.putf('g_end[{k}] = intv.b    # end of Gauss nodes', k=k)

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

        # generate Gauss weights arguments
        self.put(self.dimrep('&self.gaussweights{0}[g_sta[{0}]]') + ',')

        self.putf('self.fields[{idx}],',
                idx=self.dimrep('g_sta[{0}]:g_end[{0}]'))
        if self.num_constants > 0:
            self.put('&self.constants[0],')

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
        elif isinstance(s, vform.Parameter):
            assert False, 'parameters should not be stored in arrays'
        else:
            assert False, 'invalid source %s for var %s' % (s, var.name)

    def generate_metadata(self):
        # generate an 'inputs' class method which returns a name->shape dict
        self.put('@classmethod')
        self.put('def inputs(cls):')
        self.indent()
        self.put('return {')
        for inp in self.vform.inputs:
            self.putf("    '{name}': {shp},", name=inp.name, shp=inp.shape)
        self.put('}')
        self.dedent()
        self.put('')

        # generate an 'parameters' class method which returns a name->shape dict
        self.put('@classmethod')
        self.put('def parameters(cls):')
        self.indent()
        self.put('return {')
        for par in self.vform.params:
            self.putf("    '{name}': {shp},", name=par.name, shp=par.shape)
        self.put('}')
        self.dedent()
        self.put('')

    def generate_init(self):
        vf = self.vform

        used_spaces = sorted(set(bf.space for bf in vf.basis_funs))
        assert len(used_spaces) in (1,2), 'Number of spaces should be 1 or 2'
        assert all(sp in (0, 1) for sp in used_spaces), 'Space index should be 0 or 1'
        used_kvs = tuple('kvs%d' % sp for sp in used_spaces)
        input_args = ', '.join([inp.name for inp in vf.inputs] + [param.name for param in vf.params])

        self.putf('def __init__(self, {kvs}, {inp}{bbox}{boundary}):', kvs=', '.join(used_kvs),
                bbox = ', bbox' if self.on_demand else '',
                boundary = ', boundary' if vf.is_boundary_integral() else '',
                inp=input_args)
        self.indent()

        self.putf('self.arity = {ar}', ar=vf.arity)
        self.putf('self.nqp = max([kv.p for kv in {kvs}]) + 1', kvs=' + '.join(used_kvs))
        if len(used_spaces) == 1:
            self.put('kvs1 = kvs0')

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
            assert not vf.is_boundary_integral(), 'boundary on demand not implemented'
        else:
            if vf.is_boundary_integral():
                self.put('gaussgrid, gaussweights = make_boundary_quadrature([kv.mesh for kv in kvs0], self.nqp, boundary)')
            else:
                self.put('gaussgrid, gaussweights = make_tensor_quadrature([kv.mesh for kv in kvs0], self.nqp)')

        self.put('self.gaussgrid = gaussgrid')
        for k in range(self.dim):
            self.putf('self.gaussweights{k} = gaussweights[{k}]', k=k)

        if self.on_demand:
            self.put('self.bbox_ofs[:] = tuple(bb[0] * self.nqp for bb in bbox)')

        self.put('')

        # BaseAssembler always uses S1 for i and S0 for j for a bilinear form
        for sp in (0,1):
            self.putf('assert len(kvs{sp}) == {dim}, "Assembler requires {dim} knot vectors"', sp=sp)
            self.putf('self.S{sp}_ndofs[:] = [kv.numdofs for kv in kvs{sp}]', sp=sp)

            if vf.is_boundary_integral():
                # boundary integral has dimension 1 along the normal dimension
                self.putf('self.S{sp}_ndofs[boundary[0]] = 1', sp=sp)

            for k in range(self.dim):
                self.putf('self.S{sp}_meshsupp{k} = self.nqp * kvs{sp}[{k}].mesh_support_idx_all()', sp=sp, k=k)
                self.putf('self.S{sp}_C{k} = compute_values_derivs(kvs{sp}[{k}], gaussgrid[{k}], derivs={maxderiv})',
                        k=k, sp=sp, maxderiv=self.numderiv)

                if vf.is_boundary_integral():
                    self.putf('if boundary[0] == {k}:', k=k)
                    self.indent()
                    # restrict quadrature mesh to a single basis function over the first interval
                    self.putf('self.S{sp}_meshsupp{k} = np.arange(2, dtype=np.intp).reshape((1,2))', sp=sp, k=k)
                    self.put('if boundary[1] == 0:')
                    self.indent()
                    # restrict to first basis function and first grid point
                    self.putf('self.S{sp}_C{k} = self.S{sp}_C{k}[0:1, 0:1, :]', sp=sp, k=k)
                    self.dedent()
                    self.put('else:')
                    self.indent()
                    # restrict to last basis function and last grid point
                    self.putf('self.S{sp}_C{k} = self.S{sp}_C{k}[-1:, -1:, :]', sp=sp, k=k)
                    self.dedent()
                    self.dedent()

        self.put('')

        self.put('N = tuple(gg.shape[0] for gg in gaussgrid)  # grid dimensions')

        # initialize storage for fields and constants
        self.put('')
        self.put('# Fields:')
        for var in self.global_info:    # put some comments for readability
            _, sz, ofs = self.global_info[var]
            self.putf('#  - {var}: ofs={ofs} sz={sz}', var=var, ofs=ofs, sz=sz)

        self.putf('self.fields = np.empty(N + ({nf},))', nf=self.num_globals)
        if self.num_constants > 0:
            self.putf('self.constants = np.zeros({np})', np=self.num_constants)
            self.putf('self.update_params({prms})',
                    prms = ', '.join(par.name + '=' + par.name for par in self.vform.params))

        # declare array storage for non-global variables
        if self.vform.precomp:
            self.put('# Temp fields:')
            for var in self.temp_info:    # put some comments for readability
                _, sz, ofs = self.temp_info[var]
                self.putf('#  - {var}: ofs={ofs} sz={sz}', var=var, ofs=ofs, sz=sz)
            self.putf('cdef {typ} temp_fields = np.empty(N + ({nt},))',
                    typ=self.field_type(), nt=self.num_temp)

        # declare/initialize array variables from InputFields
        for var in vf.linear_deps:
            # exclude basis function nodes and non-InputFields
            if not isinstance(var, vform.BasisFun) and var.src and isinstance(var.src, vform.InputField):
                src = self.parse_src(var)
                if var.is_global:
                    var, sz, ofs = self.global_info[var.name]
                    arr = 'self.fields'
                else:
                    var, sz, ofs = self.temp_info[var.name]
                    arr = 'temp_fields'

                assert not var.symmetric, 'symmetric input matrices not currently supported'

                # NB: Cython (as of 0.29.20) can't assign an ndarray to a slice
                # of a typed memoryview, so we use the underlying ndarray via .base
                self.putf('{arr}.base[{dims}, {ofs}:{end}] = {src}.reshape(N + (-1,))',
                        arr=arr, dims=self.dimrep(':'), ofs=ofs, end=ofs+sz, src=src)

        if vf.precomp:
            # call precompute function
            self.putf('{classname}.precompute_fields(', classname=self.classname)
            self.indent(2)
            # generate size arguments
            self.put(self.dimrep('gaussgrid[{}].shape[0]') + ',')
            # generate Gauss weight arguments
            self.put(self.dimrep('&self.gaussweights{}[0]') + ',')
            # generate arguments for input fields
            self.put('temp_fields,')
            self.put('self.fields,')
            if self.num_constants > 0:
                self.put('&self.constants[0],')
            self.dedent(2)
            self.put(')')

        self.initialize_custom_fields()

        if vf.is_boundary_integral():
            # eliminate the unused knotvector normal to the boundary;
            # these kvs are used to determine the matrix structure in assemble_entries()
            self.put('kvs0 = kvs0[:boundary[0]] + kvs0[boundary[0]+1:]')
            self.put('kvs1 = kvs1[:boundary[0]] + kvs1[boundary[0]+1:]')
        self.put('self.kvs = (kvs0, kvs1)')

        self.end_function()

    def field_var_entry(self, var, idx, first_entry=False):
        """Generate a reference to the current entry in the given field variable."""
        I = self.dimrep(idx + '{}')
        if first_entry:
            I += ', 0'
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
        self.put('# Gauss weights')
        self.put(self.dimrep('double* _gw{}') + ',')
        self.put('# input')
        self.put(self.field_type() + ' _temp_fields,')
        self.put('# output')
        self.put(self.field_type() + ' _fields,')
        if self.num_constants > 0:
            self.put('# constant parameters')
            self.put('double constants[],')    # array for constant parameters
        self.dedent()
        self.put(') nogil:')

        # start main loop
        self.start_loop_with_fields(local_vars=vf.precomp, temp=True)

        # end main loop
        for _ in range(self.dim):
            self.code.end_loop()
        self.end_function()

    def generate_update(self):
        self.putf('def update(self, {args}):',
                args=', '.join('%s=None' % inp.name for inp in self.updatable))
        self.indent()
        self.put('N = self.fields.base.shape[:-1]')

        # declare/initialize array variables
        for var in self.vform.linear_deps:
            if not isinstance(var, vform.BasisFun) and var.src in self.updatable:
                assert var.scope == vform.Scope.FIELD and var.is_global, 'only global array vars can be updated'
                self.putf("if {name}:", name=var.src.name)
                self.indent()

                var, sz, ofs = self.global_info[var.name]
                arr = 'self.fields'
                # NB: Cython (as of 0.29.20) can't assign an ndarray to a slice
                # of a typed memoryview, so we use the underlying ndarray via .base
                self.putf('{arr}.base[{dims}, {ofs}:{end}] = {src}.reshape(N + (-1,))',
                        arr=arr, dims=self.dimrep(':'), ofs=ofs, end=ofs+sz, src=self.parse_src(var))

                self.dedent()
        self.end_function()

    def generate_update_params(self):
        self.putf('def update_params(self, {args}):',
                args=', '.join('%s=None' % par.name for par in self.vform.params))
        self.indent()

        for par in self.vform.params:
            self.putf("if {name} is not None:", name=par.name)
            self.indent()
            self.putf("if np.shape({name}) != {shp}: raise TypeError('{name} has improper shape')",
                    name=par.name, shp=par.shape)
            ofs = self.constant_info[par.name][2]
            sz  = self.constant_info[par.name][1]
            self.putf("values = np.ravel({name})", name=par.name)
            self.putf("for i in range({sz}):", sz=sz)
            self.indent()
            self.putf("self.constants[{ofs} + i] = values[i]", ofs=ofs)
            self.dedent()
            self.dedent()
        self.end_function()

    # main code generation entry point

    def generate(self):
        self.vform.finalize(do_precompute=not self.on_demand)
        self.numderiv = self.vform.find_max_deriv()

        # determine layout of global 'constants' array.
        # global constants are parameters and constants computed during precompute.
        constants = self.vform.params + [var for var in self.vform.kernel_deps
                if var.scope == vform.Scope.CONSTANT and var.expr and var.is_global]
        self.constant_info, self.num_constants = allocate_array(constants)

        # determine layout of global 'fields' array.
        # globals are either input fields or fields computed during precompute.
        global_vars = [var for var in self.vform.kernel_deps
                if var.scope == vform.Scope.FIELD and var.is_global]
        self.global_info, self.num_globals = allocate_array(global_vars)

        # collect input fields needed for precompute which are not in 'fields' yet
        self.precomp_array_deps = [var for var in self.vform.precomp
                if var.scope == vform.Scope.FIELD and var.src and not var.is_global]
        self.temp_info, self.num_temp = allocate_array(self.precomp_array_deps)

        self.env = {
            'dim': self.vform.dim,
            'geo_dim': self.vform.geo_dim,
        }
        for var in self.constant_info:
            _, sz, ofs = self.constant_info[var]
            self.putf('#  - {var}: ofs={ofs} sz={sz}', var=var, ofs=ofs, sz=sz)

        baseclass = 'BaseVectorAssembler' if self.vec else 'BaseAssembler'
        self.putf('cdef class {classname}({base}{dim}D):',
                classname=self.classname, base=baseclass)
        self.indent()

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

        if self.num_constants > 0:
            self.put('')
            self.generate_update_params()

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

# members which are shared by the scalar and vector base assembler
cdef class _CommonBase{{DIM}}D:
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
    cdef double[::1] constants
    cdef double[{{ dimrepeat(':') }}, ::1] fields
    {%- for k in range(DIM) %}
    cdef double[::1] gaussweights{{k}}
    {%- endfor %}
    # on-demand assemblers only:
    cdef size_t[{{DIM}}] bbox_ofs   # bounding box of indices which will be computed
    # vector assemblers only:
    cdef size_t[2] numcomp          # number of vector components for trial and test functions


cdef class BaseAssembler{{DIM}}D(_CommonBase{{DIM}}D):
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



cdef class BaseVectorAssembler{{DIM}}D(_CommonBase{{DIM}}D):
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

from pyiga.quadrature import make_tensor_quadrature, make_boundary_quadrature

from pyiga.assemble_tools_cy cimport (
    BaseAssembler1D, BaseAssembler2D, BaseAssembler3D,
    BaseVectorAssembler1D, BaseVectorAssembler2D, BaseVectorAssembler3D,
    IntInterval, make_intv, intersect_intervals,
)
from pyiga.assemble_tools import compute_values_derivs
from pyiga.utils import LazyCachingArray, grid_eval, grid_eval_transformed

"""
