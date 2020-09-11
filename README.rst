
.. |travis| image:: https://travis-ci.org/c-f-h/pyiga.svg?branch=master
    :target: https://travis-ci.org/c-f-h/pyiga
.. |appveyor| image:: https://ci.appveyor.com/api/projects/status/1enc32o4ts2w9w17/branch/master?svg=true
   :target: https://ci.appveyor.com/project/c-f-h/pyiga
.. |codecov| image:: https://codecov.io/gh/c-f-h/pyiga/branch/master/graph/badge.svg
  :target: https://codecov.io/gh/c-f-h/pyiga

pyiga |travis| |appveyor| |codecov|
===================================

``pyiga`` is a Python research toolbox for Isogeometric Analysis. Its current highlights are:

* Automatic generation of efficient matrix assembling code from a high-level, FEniCS_-like description of the bilinear form. See the section "Assembling custom forms" in the `Documentation`_  as well as the convection-diffusion and Navier-Stokes examples below.
* Adaptivity via HB- and THB-spline spaces and a local multigrid solver for adaptive IgA (`see the preprint <https://www.ricam.oeaw.ac.at/files/reports/19/rep19-34.pdf>`_). See `adaptive.ipynb <notebooks/adaptive.ipynb>`_ for an example.
* Fast assembling by a black-box low-rank assembling algorithm described in
  `this paper <http://dx.doi.org/10.1016/j.cma.2018.01.014>`_
  (or `this technical report <http://www.numa.uni-linz.ac.at/publications/List/2017/2017-02.pdf>`_).
* Extensive support for fast tensor approximation methods for tensor product IgA.

To find out more, have a look at the `Documentation`_ and the examples below.

Examples
--------

The ``notebooks`` directory contains several examples of how to use ``pyiga``:

*  `geometry.ipynb <notebooks/geometry.ipynb>`_: create and manipulate geometry functions
*  `solve-poisson.ipynb <notebooks/solve-poisson.ipynb>`_: solve a Poisson equation and plot the solution
*  `solve-convdiff.ipynb <notebooks/solve-convdiff.ipynb>`_: solve a convection-diffusion problem with random inclusions
*  `solve-stokes.ipynb <notebooks/solve-stokes.ipynb>`_: solve stationary Stokes flow and plot the velocity field
*  `solve-navier-stokes.ipynb <https://nbviewer.jupyter.org/github/c-f-h/pyiga/blob/master/notebooks/solve-navier-stokes.ipynb>`_: solve the instationary Navier-Stokes equations with a time-adaptive Rosenbrock integrator and produce an animation of the result
*  `adaptive.ipynb <notebooks/adaptive.ipynb>`_: an adaptive solve-estimate-mark-refine loop using a local multigrid solver.
*  `mantle-convection.ipynb <https://nbviewer.jupyter.org/gist/c-f-h/060f225465ee990faab4941a6cfd2562>`_: Rayleigh-Bénard convection


Installation
------------

``pyiga`` is compatible with Python 3.6 and higher.

Before installing, make
sure you have recent versions of **Numpy**, **Scipy** and **Cython** installed
and that your environment can compile Python extension modules.
Numpy 1.14 or higher is required.
If you do not have such an environment set up yet, the easiest way to get it
is by installing Anaconda_ (this can be done without administrator privileges).

Clone this repository and execute ::

    $ python setup.py install --user

in the main directory. The installation script should now compile the Cython
extensions and then install the package in your user directory. If you prefer
to install the package globally, skip the ``--user`` flag; this requires
administrator rights.

If you have Intel MKL installed on your machine, be sure to install the
**pyMKL** package; if ``pyiga`` detects this package, it will use the
MKL PARDISO sparse direct solver instead of the internal scipy solver
(typically SuperLU).

Updating
~~~~~~~~

If you have already installed the package and want to update to the latest
version, assuming that you have cloned it from Github, you can simply move to
the project directory and execute ::

    $ git pull
    $ python setup.py install --user

Running tests
-------------

`pyiga` comes with a small test suite to test basic functionality. Depending on
your test runner of choice, move to the main directory and execute
``nosetests`` or ``py.test`` to run the tests.

If the test runner fails to find the Cython extensions modules (``pyiga.bspline_cy`` etc.),
you may have to run ``python setup.py build_ext -i`` to build them in-place.

Usage
-----

After successful installation, you should be able to load the package. A simple example:

.. code:: python

    from pyiga import bspline, geometry, assemble

    kv = bspline.make_knots(3, 0.0, 1.0, 50)    # knot vector over (0,1) with degree 3 and 50 knot spans
    geo = geometry.quarter_annulus()            # a NURBS representation of a quarter annulus
    K = assemble.stiffness((kv,kv), geo=geo)    # assemble a stiffness matrix for the 2D tensor product
                                                # B-spline basis over the quarter annulus

There is a relatively complete `Documentation`_. Beyond that, look at the code,
the unit tests, and the `IPython notebooks`_ to learn more.


.. _IPython notebooks: ./notebooks
.. _Documentation: http://pyiga.readthedocs.io/en/latest/
.. _FEniCS: https://fenicsproject.org/
.. _Anaconda: https://www.anaconda.com/distribution/

FAQ
---

During compilation, I get an error message involving ``numpy._build_utils``.
~~~~~

Try installing/upgrading setuptools: ::

    $ pip install --upgrade --user setuptools
