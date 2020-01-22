import os.path
import argparse
from pyiga import vform
from pyiga.codegen import cython as backend

def parse_args():
    parser = argparse.ArgumentParser(description='Convert a vform expression to code for an assembler class.')
    parser.add_argument('expr', type=str, help='the expression to be compiled')
    parser.add_argument('--dim', type=int, default=2, help='the space dimension')
    parser.add_argument('--name', type=str, default='CustomAsm', help='the name of the assembler class')
    parser.add_argument('--scalarinput', nargs='*', metavar='NAME', help='names of scalar input fields')
    parser.add_argument('--vectorinput', nargs='*', metavar='NAME', help='names of vector input fields')
    parser.add_argument('--ondemand', action='store_true', help='create an on demand assembler')
    return parser.parse_args()

if __name__ == '__main__':
    _args = parse_args()

    vf = vform.VForm(dim=_args.dim)
    u, v = vf.basisfuns()

    # create scalar input fields
    if _args.scalarinput:
        for funcname in _args.scalarinput:
            locals()[funcname] = vf.input(funcname)

    # create vector input fields
    if _args.vectorinput:
        for funcname in _args.vectorinput:
            locals()[funcname] = vf.input(funcname, shape=(_args.dim,))

    vf.add(eval(_args.expr, vars(vform), locals()))

    code = backend.CodeGen()
    backend.AsmGenerator(vf, _args.name, code, on_demand=_args.ondemand).generate()

    print(backend.preamble())
    print(code.result())
