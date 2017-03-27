``pyiga``
=========

``pyiga`` is a Python research toolbox for Isogeometric Analysis.


Installation
------------

``pyiga`` is currently only compatible with Python 3. Before installing, make
sure you have recent versions of **Numpy**, **Scipy** and **Cython** installed
and that your environment can compile Python extension modules.

Clone this repository and execute ::

    $ python setup.py install --user

in the main directory. The installation script should now compile the Cython
extensions and then install the package in your user directory. If you prefer
to install the package globally, skip the ``--user`` flag; this requires
administrator rights.

Running tests
-------------

`pyiga` comes with a small test suite to test basic functionality. Depending on
your test runner of choice, move to the main directory and execute
``nosetests`` or ``py.test`` to run the tests.

Usage
-----

After successful installation, you should be able to load the package. A simple example::

    from pyiga import bspline, geometry, assemble

    kv = bspline.make_knots(3, 0.0, 1.0, 50)    # knot vector over (0,1) with degree 3 and 50 knot spans
    geo = geometry.bspline_quarter_annulus()    # a B-spline approximation of a quarter annulus
    K = assemble.stiffness((kv,kv), geo=geo)    # assemble a stiffness matrix for the tensor product B-spline
                                                # basis over the given geometry

Right now there is now comprehensive documentation, so look at the code, the unit tests,
and the IPython notebooks to learn more.

