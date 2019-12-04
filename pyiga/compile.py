import pyiga
from pyiga.codegen import cython as codegen

import tempfile
import importlib
import sys
import os.path
import appdirs
import hashlib

import numpy

import distutils
from distutils.core import Extension

import Cython
import Cython.Compiler.Options
from Cython.Build.Inline import _get_build_extension
from Cython.Build.Dependencies import cythonize


PYIGAPATH = os.path.normpath(os.path.join(os.path.split(pyiga.__file__)[0], '..'))
MODDIR = os.path.join(appdirs.user_cache_dir('pyiga'), 'modules')


def _compile_cython_module_nocache(src, modname, verbose=False):
    modfile = os.path.join(MODDIR, modname + '.pyx')
    with open(modfile, 'w+') as f:
        f.write(src)
    Cython.Compiler.Options.cimport_from_pyx = True

    include_dirs = [
        numpy.get_include()
    ]
    extra_compile_args = ['-O3', '-march=native', '-ffast-math', '-fopenmp']

    extension = Extension(name=modname,
                          language="c++",
                          sources=[modfile],
                          include_dirs=include_dirs,
                          extra_compile_args=extra_compile_args)

    cython_include_dirs = [PYIGAPATH]
    build_extension = _get_build_extension()
    build_extension.extensions = cythonize([extension],
                                           include_path=cython_include_dirs,
                                           compiler_directives={'language_level': 3},
                                           quiet=False)
    build_extension.build_temp = MODDIR
    build_extension.build_lib = MODDIR

    distutils.log.set_verbosity(verbose)
    build_extension.run()
    return importlib.import_module(modname)


def compile_cython_module(src, verbose=False):
    """Compile module from the given Cython source string and return the loaded
    module object.

    Performs caching."""
    os.makedirs(MODDIR, exist_ok=True)
    if not MODDIR in sys.path:
        sys.path.append(MODDIR)

    # NB: builtin hash() is not deterministic across runs! use SHAKE instead
    modname = 'mod' + hashlib.shake_128(src.encode()).hexdigest(8)
    try:
        mod = importlib.import_module(modname)
    except ImportError:
        mod = _compile_cython_module_nocache(src, modname, verbose=verbose)
    return mod


def generate(vf, classname='CustomAssembler'):
    code = codegen.CodeGen()
    codegen.AsmGenerator(vf, classname, code).generate()
    return codegen.preamble() + '\n' + code.result()

def compile_vform(vf, verbose=False):
    src = generate(vf)
    mod = compile_cython_module(src, verbose=verbose)
    return mod.CustomAssembler

def compile_vforms(vfs, verbose=False):
    vfs = tuple(vfs)
    n = len(vfs)
    names = tuple('CustomAssembler%d' % i for i in range(n))

    code = codegen.CodeGen()
    for (name, vf) in zip(names, vfs):
        codegen.AsmGenerator(vf, name, code).generate()
    src = codegen.preamble() + '\n' + code.result()

    mod = compile_cython_module(src, verbose=verbose)
    return tuple(getattr(mod, name) for name in names)
