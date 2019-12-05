from pyiga.codegen import cython as codegen
from pyiga import vform

def my_stiffness_vf(dim):
    # same as stiffness_vf(), but slower implementation
    from pyiga.vform import VForm, inner, grad, dx
    V = VForm(dim)
    u, v = V.basisfuns()
    V.add(inner(grad(u), grad(v)) * dx)
    return V

def test_codegen_poisson2d():
    code = codegen.CodeGen()
    vf = vform.stiffness_vf(2)
    codegen.AsmGenerator(vf, 'TestAsm', code).generate()
    code = codegen.preamble() + '\n' + code.result()

def test_codegen_poisson3d():
    code = codegen.CodeGen()
    vf = my_stiffness_vf(3)
    codegen.AsmGenerator(vf, 'TestAsm', code).generate()
    code = codegen.preamble() + '\n' + code.result()
