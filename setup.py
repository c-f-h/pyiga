from setuptools import setup
from setuptools.extension import Extension
from Cython.Build import cythonize
import numpy

USE_OPENMP = True

c_args = ['-O3', '-march=native', '-ffast-math']

if USE_OPENMP:
    c_args_openmp = l_args_openmp = ['-fopenmp']
else:
    c_args_openmp = l_args_openmp = []

extensions = [
    Extension("pyiga.bspline_cy",
             ["pyiga/bspline_cy.pyx"],
        include_dirs = [numpy.get_include()],
        extra_compile_args=c_args,
    ),
    #Extension("pyiga.lowrank_cy",
    #         ["pyiga/lowrank_cy.pyx"],
    #    include_dirs = [numpy.get_include()],
    #    extra_compile_args=c_args,
    #),
    Extension("pyiga.mlmatrix_cy",
             ["pyiga/mlmatrix_cy.pyx"],
        include_dirs = [numpy.get_include()],
        extra_compile_args=c_args + c_args_openmp,
        extra_link_args=l_args_openmp,
    ),
    Extension("pyiga.assemble_tools_cy",
             ["pyiga/assemble_tools_cy.pyx"],
        include_dirs = [numpy.get_include()],
        language='c++',
        extra_compile_args=c_args + c_args_openmp,
        extra_link_args=l_args_openmp,
        #define_macros=[('CYTHON_TRACE', '1')]
    ),
    Extension("pyiga.fast_assemble_cy",
             ["pyiga/fastasm.cc",
              "pyiga/fast_assemble_cy.pyx"],
        include_dirs = [numpy.get_include()],
        language='c++',
        extra_compile_args=c_args,
    ),
]


setup(
    name = 'pyiga',
    version = '0.0.0',
    description = 'A Python research toolbox for Isogeometric Analysis',
    author = 'Clemens Hofreither',
    author_email = 'chofreither@numa.uni-linz.ac.at',
    url = 'https://github.com/c-f-h/pyiga',

    classifiers=[
        'Programming Language :: Python :: 3',
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Mathematics',
        'License :: Free For Educational Use',
    ],
    packages = ['pyiga'],

    ext_modules = cythonize(extensions),
    package_data = {
        'pyiga': [ '*.pyx' , '*.pxd' , '*.pxi' ,],
    },

    setup_requires = ['numpy', 'Cython'],
    install_requires = [
        'numpy',
        'scipy',
        'future;python_version<"3.0"',
        'futures;python_version<"3.0"'   # backport of concurrent.futures to Py2
    ],

    tests_require = 'nose',
    test_suite = 'nose.collector'
)
