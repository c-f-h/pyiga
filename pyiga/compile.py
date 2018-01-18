import pyiga

import tempfile
import importlib
import sys
import os.path
import appdirs

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
    extra_compile_args = []

    extension = Extension(name=modname,
                          language="c++",
                          sources=[modfile],
                          include_dirs=include_dirs,
                          extra_compile_args=extra_compile_args)

    cython_include_dirs = [PYIGAPATH]
    build_extension = _get_build_extension()
    build_extension.extensions = cythonize([extension],
                                           include_path=cython_include_dirs,
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

    modname = 'mod' + hex(abs(hash(src)))[2:]
    try:
        mod = importlib.import_module(modname)
    except ImportError:
        mod = _compile_cython_module_nocache(src, modname, verbose=verbose)
    return mod
