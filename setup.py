from setuptools import setup
from setuptools.extension import Extension
from Cython.Build import cythonize
import numpy

extensions = [
    Extension("pyiga.bspline_cy",
             ["pyiga/bspline_cy.pyx"],
        include_dirs = [numpy.get_include()],
        extra_compile_args=['-O3'],
    ),
    #Extension("pyiga.lowrank_cy",
    #         ["pyiga/lowrank_cy.pyx"],
    #    include_dirs = [numpy.get_include()],
    #    extra_compile_args=['-O3'],
    #),
    Extension("pyiga.mlmatrix_cy",
             ["pyiga/mlmatrix_cy.pyx"],
        include_dirs = [numpy.get_include()],
        extra_compile_args=['-O3'],
    ),
    Extension("pyiga.assemble_tools_cy",
             ["pyiga/assemble_tools_cy.pyx"],
        include_dirs = [numpy.get_include()],
        language='c++',
        extra_compile_args=['-O3'],
        #define_macros=[('CYTHON_TRACE', '1')]
    ),
    Extension("pyiga.fast_assemble_cy",
             ["pyiga/fastasm.cc",
              "pyiga/fast_assemble_cy.pyx"],
        include_dirs = [numpy.get_include()],
        language='c++',
        extra_compile_args=['-O3'],
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
        'pyiga': [ '*.pyx' , '*.pxd' ],
    },

    test_suite = 'nose.collector'
)
