import pyiga
from pyiga.codegen import cython as codegen

import tempfile
import importlib
import sys
import os.path
import platformdirs
import hashlib

import numpy

from setuptools import Extension

import Cython
import Cython.Compiler.Options
from Cython.Build.Inline import _get_build_extension
from Cython.Build.Dependencies import cythonize


PYIGAPATH = os.path.normpath(os.path.join(os.path.split(pyiga.__file__)[0], '..'))
MODDIR = os.path.join(platformdirs.user_cache_dir('pyiga'), 'modules')


def _compile_cython_module_nocache(src, modname, verbose=False):
    modfile = os.path.join(MODDIR, modname + '.pyx')
    with open(modfile, 'w+') as f:
        f.write(src)
    Cython.Compiler.Options.cimport_from_pyx = True

    include_dirs = [
        numpy.get_include()
    ]
    extra_compile_args = ['-O3', '-march=native', '-ffast-math', '-fopenmp', '-g1']

    c_macros = [("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")]

    extension = Extension(name=modname,
                          sources=[modfile],
                          include_dirs=include_dirs,
                          extra_compile_args=extra_compile_args,
                          define_macros=c_macros)

    cython_include_dirs = [PYIGAPATH]
    build_extension = _get_build_extension()
    build_extension.extensions = cythonize([extension],
                                           include_path=cython_include_dirs,
                                           compiler_directives={'language_level': 3},
                                           quiet=False)
    build_extension.build_temp = MODDIR
    build_extension.build_lib = MODDIR

    #distutils.log.set_verbosity(verbose)
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


def generate(vf, classname='CustomAssembler', on_demand=False):
    """Generate Cython code for an assembler class which implements the vform `vf`."""
    code = codegen.CodeGen()
    codegen.AsmGenerator(vf, classname, code, on_demand=on_demand).generate()
    return codegen.preamble() + '\n' + code.result()

# There are two levels of caching for compiling vforms: (1) from a hash of the
# vform expression to the assembler class, (2) from a hash of the Cython source
# code to the compiled and loaded extension module.
#
# (2) is the most important one since compiling Cython -> C -> extension module
# is very slow. This cache persists across processes since the modules are
# stored on disk.
#
# (1) can still be useful since going from VForm to Cython source code, while
# much faster than the compilation steps afterwards, can still add up if
# compile_vform() is called repeatedly (like in an adaptive refinement loop).
# This cache (which uses the following __vform_asm_cache dict) is much lighter-
# weight and is only kept in-process.
#
# As an added benefit, this cache allows us to cache compilers for predefined
# vforms which are contained in the assemblers module.
#
__vform_asm_cache = dict()

def __asm_cache_args(on_demand):
    return (on_demand,)

def __add_to_vform_asm_cache(vf, asm):
    cache_key = (vf.hash(), __asm_cache_args(False))
    __vform_asm_cache[cache_key] = asm

# add predefined assemblers to the cache
from . import assemblers, vform
for dim in (2, 3):
    nD = str(dim) + 'D'
    __add_to_vform_asm_cache(vform.mass_vf(dim), getattr(assemblers, 'MassAssembler'+nD))
    __add_to_vform_asm_cache(vform.stiffness_vf(dim), getattr(assemblers, 'StiffnessAssembler'+nD))
    __add_to_vform_asm_cache(vform.heat_st_vf(dim), getattr(assemblers, 'HeatAssembler_ST'+nD))
    __add_to_vform_asm_cache(vform.wave_st_vf(dim), getattr(assemblers, 'WaveAssembler_ST'+nD))
    __add_to_vform_asm_cache(vform.divdiv_vf(dim), getattr(assemblers, 'DivDivAssembler'+nD))
    __add_to_vform_asm_cache(vform.L2functional_vf(dim), getattr(assemblers, 'L2FunctionalAssembler'+nD))
    __add_to_vform_asm_cache(vform.L2functional_vf(dim, physical=True), getattr(assemblers, 'L2FunctionalAssemblerPhys'+nD))

def compile_vform(vf, verbose=False, on_demand=False):
    """Compile the vform `vf` into an assembler class."""
    cache_key = (vf.hash(), __asm_cache_args(on_demand))
    global __vform_asm_cache
    cached_asm = __vform_asm_cache.get(cache_key)
    if cached_asm:
        return cached_asm
    else:
        src = generate(vf, on_demand=on_demand)
        mod = compile_cython_module(src, verbose=verbose)
        asm = mod.CustomAssembler
        __vform_asm_cache[cache_key] = asm
        return asm

def compile_vforms(vfs, verbose=False):
    """Compile a list of vforms into assembler classes.

    This may be faster than compiling each vform individually since they are
    all combined into one source file.
    """
    vfs = tuple(vfs)
    names = tuple('CustomAssembler%d' % i for i in range(len(vfs)))

    code = codegen.CodeGen()
    for (name, vf) in zip(names, vfs):
        codegen.AsmGenerator(vf, name, code).generate()
    src = codegen.preamble() + '\n' + code.result()

    mod = compile_cython_module(src, verbose=verbose)
    return tuple(getattr(mod, name) for name in names)
