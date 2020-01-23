import os.path
import argparse
from pyiga import vform
from pyiga.codegen import cython as backend

def parse_args():
    parser = argparse.ArgumentParser(description='Convert a vform expression to code for an assembler class.')
    parser.add_argument('expr', type=str, help='the expression to be compiled')
    parser.add_argument('--dim', type=int, default=2, help='the space dimension')
    parser.add_argument('--name', type=str, default='CustomAsm', help='the name of the assembler class')
    parser.add_argument('-o', '--output', help='the file to write to; by default, write to stdout')
    parser.add_argument('--scalarinput', nargs='*', metavar='NAME', help='names of scalar input fields')
    parser.add_argument('--vectorinput', nargs='*', metavar='NAME', help='names of vector input fields')
    parser.add_argument('--ondemand', action='store_true', help='create an on demand assembler')
    parser.add_argument('--dumptree', action='store_true', help='write the expression tree to stdout')
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

    e = eval(_args.expr, vars(vform), locals())
    if _args.dumptree:
        vform.tree_print(e)
    vf.add(e)

    code = backend.CodeGen()
    backend.AsmGenerator(vf, _args.name, code, on_demand=_args.ondemand).generate()

    f = open(_args.output, 'w') if _args.output else None
    print(backend.preamble(), file=f)
    print(code.result(), file=f)
    if _args.output:
        f.close()
