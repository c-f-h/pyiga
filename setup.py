from setuptools import setup
from setuptools.extension import Extension
from Cython.Build import cythonize
import numpy

import Cython.Compiler.Options
Cython.Compiler.Options.cimport_from_pyx = True
Cython.Compiler.Options.annotate = True

USE_OPENMP = True

c_args = ['-O3', '-march=native', '-ffast-math']
c_macros = [("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")]

if USE_OPENMP:
    c_args_openmp = l_args_openmp = ['-fopenmp']
else:
    c_args_openmp = l_args_openmp = []

extensions = [
    Extension("pyiga.geometry_cy",
             ["pyiga/geometry_cy.pyx"],
        include_dirs = [numpy.get_include()],
        extra_compile_args=c_args,
        define_macros=c_macros,
    ),
    Extension("pyiga.ieti_cy",
             ["pyiga/ieti_cy.pyx"],
        include_dirs = [numpy.get_include()],
        extra_compile_args=c_args,
        language="c++",
        define_macros=c_macros,
    ),
    Extension("pyiga.adaptive_cy",
             ["pyiga/adaptive_cy.pyx"],
        include_dirs = [numpy.get_include()],
        extra_compile_args=c_args,
        define_macros=c_macros,
    ),
    Extension("pyiga.algebra_cy",
             ["pyiga/algebra_cy.pyx"],
        include_dirs = [numpy.get_include()],
        extra_compile_args=c_args,
        language="c++",
        define_macros=c_macros,
    ),
    Extension("pyiga.assemble_cy",
             ["pyiga/assemble_cy.pyx"],
        include_dirs = [numpy.get_include()],
        extra_compile_args=c_args,
        language="c++",
        define_macros=c_macros,
    ),
    Extension("pyiga.bspline_cy",
             ["pyiga/bspline_cy.pyx"],
        include_dirs = [numpy.get_include()],
        extra_compile_args=c_args,
        language="c++",
        define_macros=c_macros,
    ),
    Extension("pyiga.lowrank_cy",
             ["pyiga/lowrank_cy.pyx"],
        include_dirs = [numpy.get_include()],
        extra_compile_args=c_args,
        define_macros=c_macros,
    ),
    Extension("pyiga.mlmatrix_cy",
             ["pyiga/mlmatrix_cy.pyx"],
        include_dirs = [numpy.get_include()],
        extra_compile_args=c_args + c_args_openmp,
        extra_link_args=l_args_openmp,
        define_macros=c_macros,
    ),
    Extension("pyiga.assemble_tools_cy",
             ["pyiga/assemble_tools_cy.pyx"],
        include_dirs = [numpy.get_include()],
        extra_compile_args=c_args + c_args_openmp,
        extra_link_args=l_args_openmp,
        define_macros=c_macros,
    ),
    Extension("pyiga.assemblers",
             ["pyiga/assemblers.pyx"],
        include_dirs = [numpy.get_include()],
        extra_compile_args=c_args + c_args_openmp,
        extra_link_args=l_args_openmp,
        define_macros=c_macros,
    ),
    Extension("pyiga.fast_assemble_cy",
             ["pyiga/fastasm.cc",
              "pyiga/fast_assemble_cy.pyx"],
        include_dirs = [numpy.get_include()],
        language='c++',
        extra_compile_args=c_args,
        define_macros=c_macros,
    ),
    Extension("pyiga.relaxation_cy",
             ["pyiga/relaxation_cy.pyx"],
        extra_compile_args=c_args,
        define_macros=c_macros,
    ),
]


setup(
    name = 'pyiga',
    version = '0.1.0',
    description = 'A Python research toolbox for Isogeometric Analysis',
    long_description = 'pyiga is a Python research toolbox for Isogeometric Analysis.\n\nPlease visit the project homepage on Github to learn more.',
    author = 'Clemens Hofreither',
    author_email = 'chofreither@ricam.oeaw.ac.at',
    url = 'https://github.com/c-f-h/pyiga',

    classifiers=[
        'Programming Language :: Python :: 3',
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Scientific/Engineering :: Physics',
        'License :: Free For Educational Use',
    ],
    packages = ['pyiga', 'pyiga.codegen'],

    ext_modules = cythonize(extensions, compiler_directives={'language_level': 3, 'legacy_implicit_noexcept': True}, annotate=True),
    package_data = {
        'pyiga': [ '*.pyx' , '*.pxd' , '*.pxi' ,],
    },

    setup_requires = ['numpy', 'Cython'],
    install_requires = [
        'Cython',
        'numpy>=1.11',
        'scipy',
        'platformdirs',
        'networkx',
        'jinja2',
        'setuptools',
    ],
)