from pyiga.codegen import cython as codegen
from pyiga import vform

def test_codegen_poisson2d():
    code = codegen.CodeGen()
    vf = vform.stiffness_vf(2)
    codegen.AsmGenerator(vf, 'TestAsm', code).generate()
    code = codegen.preamble() + '\n' + code.result()
