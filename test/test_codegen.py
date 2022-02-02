from pyiga.codegen import cython as codegen
from pyiga import vform

def my_stiffness_vf(dim):
    # same as stiffness_vf(), but slower implementation
    from pyiga.vform import VForm, inner, grad, dx
    V = VForm(dim)
    u, v = V.basisfuns()
    V.add(inner(grad(u), grad(v)) * dx)
    return V

def vector_laplace_vf(dim):
    from pyiga.vform import VForm, inner, grad, dx
    V = VForm(dim)
    u, v = V.basisfuns(components=(dim,dim))
    V.add(inner(grad(u), grad(v)) * dx)
    return V

def vector_L2functional_vf(dim):
    from pyiga.vform import VForm, inner, dx
    V = VForm(dim, arity=1)
    u = V.basisfuns(components=(dim,))
    f = V.input('f', shape=(dim,))
    V.add(inner(u, f) * dx)
    return V


def test_codegen_poisson2d():
    code = codegen.CodeGen()
    vf = vform.stiffness_vf(2)
    assert (not vf.vec) and vf.arity == 2
    codegen.AsmGenerator(vf, 'TestAsm', code).generate()
    code = codegen.preamble() + '\n' + code.result()

def test_codegen_poisson3d():
    code = codegen.CodeGen()
    vf = my_stiffness_vf(3)
    assert (not vf.vec) and vf.arity == 2
    codegen.AsmGenerator(vf, 'TestAsm', code).generate()
    code = codegen.preamble() + '\n' + code.result()

def test_codegen_vectorlaplace2d():
    code = codegen.CodeGen()
    vf = vector_laplace_vf(2)
    assert vf.vec == 2*2 and vf.arity == 2
    codegen.AsmGenerator(vf, 'TestAsm', code).generate()
    code = codegen.preamble() + '\n' + code.result()

def test_codegen_functional():
    code = codegen.CodeGen()
    vf = vform.L2functional_vf(3, updatable=True)
    assert (not vf.vec) and vf.arity == 1
    codegen.AsmGenerator(vf, 'TestAsm', code).generate()
    code = codegen.preamble() + '\n' + code.result()

def test_codegen_vecfunctional():
    code = codegen.CodeGen()
    vf = vector_L2functional_vf(3)
    assert vf.vec == 3 and vf.arity == 1
    codegen.AsmGenerator(vf, 'TestAsm', code).generate()
    code = codegen.preamble() + '\n' + code.result()

def test_codegen_parameter():
    from pyiga.vform import VForm, inner, grad, dx, norm
    code = codegen.CodeGen()
    dim = 2
    vf = VForm(dim, arity=1)
    u = vf.basisfuns()
    a = vf.parameter('a')
    b = vf.parameter('b', shape=(dim,))
    vf.add(norm(a * b) * inner(grad(u), b / norm(a * b)) * dx)
    codegen.AsmGenerator(vf, 'TestAsm', code).generate()
    code = codegen.preamble() + '\n' + code.result()
    print(code)

def test_codegen_wave_st2d():
    code = codegen.CodeGen()
    vf = vform.wave_st_vf(2)
    assert (not vf.vec) and vf.arity == 2 and vf.spacetime
    codegen.AsmGenerator(vf, 'TestAsm', code).generate()
    code = codegen.preamble() + '\n' + code.result()
