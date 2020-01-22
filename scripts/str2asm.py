import os.path
import argparse
from pyiga import vform
from pyiga.codegen import cython as backend

def str2vf(expr, dim=2):
    vf = vform.VForm(dim=dim)
    u, v = vf.basisfuns()
    vf.add(eval(expr, vars(vform), locals()))
    return vf

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert a vform expression to code for an assembler class.')
    parser.add_argument('expr', type=str, help='the expression to be compiled')
    parser.add_argument('--name', type=str, default='CustomAsm', help='the name of the assembler class')
    parser.add_argument('--ondemand', action='store_true', help='create an on demand assembler')
    args = parser.parse_args()

    vf = str2vf(args.expr)
    code = backend.CodeGen()
    backend.AsmGenerator(vf, args.name, code, on_demand=args.ondemand).generate()

    print(backend.preamble())
    print(code.result())
