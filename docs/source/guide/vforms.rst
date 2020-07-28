Assembling custom forms
=======================

``pyiga`` makes it possible to define custom linear or bilinear forms and have highly
efficient code for assembling them automatically generated, similar to
the facilities in `FEniCS <https://fenicsproject.org/>`__ or
`NGSolve <https://ngsolve.org/>`__.

.. _sec-stringbased:

Assembling variational forms from their textual description
-----------------------------------------------------------

The most convenient interface for assembling custom forms is by calling
:func:`pyiga.assemble.assemble` with a string describing the problem as the
first argument. Here is a simple but complete example::

    from pyiga import bspline, geometry, assemble

    kvs = (bspline.make_knots(3, 0.0, 1.0, 30),
           bspline.make_knots(3, 0.0, 1.0, 30))
    geo = geometry.quarter_annulus()
    A = assemble.assemble('inner(grad(u), grad(v)) * dx', kvs, geo=geo)

Here we define a cubic 2D tensor product spline space with 30 intervals per
coordinate direction and then assemble a standard Laplace variational form over
a quarter annulus geometry. The result is the sparse stiffness matrix `A`.

The string representing the integrand occurring in this example should be
mostly self-explanatory: it represents the inner product of the gradients of
the trial and test functions `u` and `v`, multiplied by the volume measure
`dx`.

In a similar way we can assemble linear functionals, such as we would need for
computing a right-hand side for a discretized problem::

    def f(x, y): return np.cos(x) * np.exp(y)
    rhs = assemble.assemble('f * v * dx', kvs, geo=geo, f=f)

Note that we specified the additional input function `f` as a keyword argument
to :func:`.assemble`. The :func:`.assemble` function automatically determined
that we are assembling a linear functional from the fact that we are referring
to only one basis function, `v`, in the description of our form. The result is
not a 1D vector as one might expect, but a 2D array whose shape is ::

    rhs.shape == (kvs[0].numdofs, kvs[1].numdofs)

However, it is easy to convert it to a 1D vector by calling :code:`rhs.ravel()`.

Vector-valued problems are realized by informing :func:`.assemble` of the
number of components of the basis functions via the `bfuns` argument. For
instance, the stiffness matrix for the vector Laplace problem could be realized
as follows::

    A = assemble.assemble('inner(grad(u), grad(v)) * dx', kvs,
            bfuns=[('u',2), ('v',2)], geo=geo)

The :func:`.assemble` function also works in hierarchical spline spaces; simply
pass an instance of :class:`pyiga.hierarchical.HSpace` as the second argument
instead of the tuple of knot vectors `kvs`. The result will be a sparse matrix
or a 1D vector, and in each case the degrees of freedom are ordered according
to the canonical order defined in the :mod:`pyiga.hierarchical` module.

For most problems, this simple string-based interface is sufficient to generate
all required matrices and vectors. However, in some edge cases it may be
necessary to define a custom vform by hand. The remainder of this document
describes the internals of how to create such objects. It also gives a more
formal description of how the expressions occurring in the strings above should
be interpreted. In essence, these strings are nothing but Python expressions
evaluated in a context where the members of the :mod:`pyiga.vform` module are
made available.

.. py:currentmodule:: pyiga.vform

Programmatically defining VForms
--------------------------------

The :mod:`pyiga.vform` module contains the tools for describing variational
forms, and :class:`pyiga.vform.VForm` is the main class used to represent
abstract variational forms. Its constructor has one required argument
which describes the space dimension. For instance, to initialize a
:class:`VForm` for a three-dimensional problem::

    from pyiga import vform

    vf = vform.VForm(3)

In order to create expressions for our form, we need objects which
represent the functions that our form operates on. By default :class:`VForm`
assumes a bilinear form, and therefore we can obtain objects for the
trial and the test function using :meth:`VForm.basisfuns` like this::

    u, v = vf.basisfuns()

The objects that we work with when specifying vforms are abstract
expressions (namely, instances of :class:`Expr`) and all have certain
properties such as a shape, :attr:`Expr.shape`, which is a tuple of
dimensions just like a numpy array has.
By default, :class:`VForm` assumes a scalar-valued problem, and therefore both
the trial function ``u`` and the test function ``v`` are scalar::

    >>> u.shape
    ()
    >>> v.shape
    ()

We can now start building expressions using these functions. Let’s first
import some commonly needed functions from the :mod:`.vform` module. ::

    from pyiga.vform import grad, div, inner, dx

We will often need the gradient of a function, obtained via :func:`grad`::

    >>> gu = grad(u)
    >>> gu.shape
    (3,)

Note that ``grad(u)`` is itself an expression.
As expected, the gradient of a scalar function is a three-component
vector. We could take the divergence (:func:`div`) of the gradient and get back a
scalar expression which represents the Laplacian
:math:`\Delta u = \operatorname{div} \nabla u` of ``u``::

    >>> Lu = div(gu)
    >>> Lu.shape
    ()

However, in order to express the standard variational form for the
Laplace problem, we only require the inner product
:math:`\nabla u \cdot \nabla v` of the gradients of our input functions,
which can be computed using :func:`inner`::

    >>> x = inner(grad(u), grad(v))
    >>> x.shape
    ()

Again, this is a scalar since :func:`inner` represents a contraction over
all axes of its input tensors; for vectors, it is the scalar product,
and for matrices, it is the Frobenius product.

.. note::
    In general, the syntax for constructing forms sticks as closely as possible to
    that of the UFL language used in FEniCS, and therefore the `UFL documentation`_
    is also a helpful resource.

.. _UFL documentation: https://readthedocs.org/projects/fenics-ufl/downloads/pdf/latest/

Finally we want to represent the integral of this expression over the
computational domain. We do this by multiplying with the symbol :data:`dx`::

    integral = inner(grad(u), grad(v)) * dx

Internally, :data:`dx` is actually a scalar expression which represents the
absolute value of the determinant of the geometry Jacobian, i.e., the scalar term
that stems from transforming the integrand from the physical domain to
the parameter domain.

We are now ready to add this expression to our :class:`VForm` via
:meth:`VForm.add`, and since the expression is rather simple, we can skip all
the intermediate steps and variables and simply do ::

    vf.add(inner(grad(u), grad(v)) * dx)

Note that the expression passed to :meth:`VForm.add` here is exactly the string
we passed to :func:`.assemble` in the first example in :ref:`the previous section
<sec-stringbased>`.

A simple example
----------------

It’s usually convenient to define vforms in their own functions so that
we don’t pollute the global namespace with the definitions from the
:mod:`.vform` module. The Laplace variational form

.. math::

    a(u,v) = \int_\Omega \nabla u \cdot \nabla v \, dx

would be defined like this::

    def laplace_vf(dim):
        from pyiga.vform import VForm, grad, inner, dx
        vf = VForm(dim)
        u, v = vf.basisfuns()
        vf.add(inner(grad(u), grad(v)) * dx)
        return vf

Calling this function results in a :class:`VForm` object::

    >>> laplace_vf(2)
    <pyiga.vform.VForm at 0x7f0fdcf5c0f0>


.. note::
    Currently, the predefined Laplace variational form in ``pyiga`` is defined
    in a different way which leads to a slightly higher performance of the
    generated code.


Vector-valued problems
----------------------

By default, basis functions are assumed to be scalar-valued. To generate
forms with vector-valued functions, simply pass the ``components``
argument with the desired sizes to :meth:`VForm.basisfuns`::

    >>> vf = vform.VForm(2)
    >>> u, v = vf.basisfuns(components=(2,2))

    >>> u.shape, v.shape
    ((2,), (2,))

We can still compute gradients (Jacobians) using :func:`grad` as before::

    >>> grad(u).shape
    (2, 2)

As a simple example, the div-div bilinear form
:math:`a(u,v) = \int_\Omega \operatorname{div} u \operatorname{div} v \,dx`
would be implemented using ::

    vf.add(div(u) * div(v) * dx)

It is also possible to mix vector and scalar functions, e.g. for
Stokes-like problems::

    vf = vform.VForm(2)
    u, p = vf.basisfuns(components=(2,1))

    vf.add(div(u) * p * dx)

In this example, ``u`` is a vector-valued function and ``p`` is scalar-valued.


Working with coefficient functions
----------------------------------

Often you will need to provide additional functions as input to your assembler,
for instance to represent a diffusion coefficient which varies over the
computational domain.  A scalar input field is declared using the
:meth:`VForm.input` method as follows::

    >>> vf = vform.VForm(2)
    >>> coeff = vf.input('coeff')

    >>> coeff.shape
    ()

The new variable ``coeff`` now represents a scalar expression that we
can work with just as with the basis functions, e.g. ::

    >>> grad(coeff).shape
    (2,)

As a simple example, to use this as a scalar diffusion coefficient, we
would do ::

    u, v = vf.basisfuns()
    vf.add(coeff * inner(grad(u), grad(v)) * dx)

Input fields can be declared vector- or matrix-valued simply by prescribing
their shape. In this example, we declare a 2x2 matrix-valued coefficient
function::

    >>> vf = vform.VForm(2)
    >>> coeff = vf.input('coeff', shape=(2,2))

    >>> coeff.shape
    (2, 2)

To actually supply these functions to the assembler, they must be passed
as keyword arguments to the constructor of the generated assembler
class. It is possible to pass either standard Python functions (in which
case differentiation is not supported) or instances of
:class:`pyiga.bspline.BSplineFunc` or :class:`pyiga.geometry.NurbsFunc`. In fact,
the predefined input ``geo`` for the geometry map is simply declared as
a vector-valued input field.
See the section :ref:`sec-compiling` for an example of how to
pass these functions.

By default, input functions are considered to be defined in the coordinates of
the parameter domain. If your input function is given in terms of physical
coordinates, declare it as follows::

    coeff = vf.input('coeff', physical=True)

.. note::
    In the simple string-based interface described in :ref:`the first section
    <sec-stringbased>`, functions passed as :class:`.BSplineFunc` or similar
    objects are assumed to be given in parametric coordinates, whereas standard
    Python functions are assumed to be given in physical coordinates.  This
    simple heuristic usually produces the expected result.

For performance reasons, it is sometimes beneficial to be able to update
a single input function without recreating the entire assembler class,
for instance when assembling the same form many times with different
coefficients in a Newton iteration for a nonlinear problem. In this
case, we can declare the function as follows::

    func = vf.input('func', updatable=True)

The generated assembler class then has an ``update()`` method which
takes the function as a keyword argument and updates it accordingly,
e.g., ::

   asm.update(func=F)


Defining constant values
------------------------

If a needed coefficient function is constant, it is unnecessary to use
the :meth:`VForm.input` machinery. Instead, we can simply define
constant values using the :func:`as_expr`, :func:`as_vector`, and
:func:`as_matrix` functions as follows::

    >>> coeff = vform.as_expr(5)
    >>> coeff.shape
    ()

    >>> vcoeff = vform.as_vector([2,3,4])
    >>> vcoeff.shape
    (3,)

    >>> mcoeff = vform.as_matrix([[2,1],[1,2]])
    >>> mcoeff.shape
    (2, 2)

We can then work with these constants exactly as with any other expression.

For constant scalar values as well as tuples of constants or expressions, the
coercion to expressions is performed implicitly, making :func:`as_expr` and
:func:`as_vector` unnecessary in these cases. This means that we can directly
write expressions such as ::

    vf.add(inner(3 * grad(u), grad(v)) * dx)
    vf.add(inner((2.0, 3.0), grad(u)) * dx)

The first example also shows that multiplication of a scalar with a vector works as
expected, i.e., the vector is multiplied componentwise with the scalar.

Defining linear (unary) forms
-----------------------------

By default, :class:`VForm` assumes the case of a bilinear form, i.e.,
having a trial function ``u`` and a test function ``v``. For defining
right-hand sides, we usually need linear forms which have only one
argument. We can do this by passing ``arity=1`` to the :class:`VForm`
constructor. The :meth:`VForm.basisfuns` method returns only a single
basis function in this case.

Below is a simple example for defining the linear form
:math:`\langle F,v \rangle = \int_\Omega f v \,dx` with a user-specified
input function ``f``::

    vf = vform.VForm(3, arity=1)
    v = vf.basisfuns()
    f = vf.input('f')
    vf.add(f * v * dx)


.. _sec-parametric:

Working with parametric derivatives
-----------------------------------

By default, a :class:`VForm` assumes that you will provide it with a geometry
map under the input field name ``geo`` and automatically transforms the
derivatives of the basis functions ``u`` and ``v``, as well as gradients of any
input fields, accordingly.
If for some reason you need to work with untransformed gradients with
respect to parametric coordinates, you can simply pass the keyword
argument ``parametric=True`` to the derivative functions such as :func:`grad`
like this::

    vf = vform.VForm(2)
    u, v = vf.basisfuns()
    f = vf.input('f')
    gu = grad(u, parametric=True)
    gf = Dx(f, 1, parametric=True)

You can compute both parametric and physical derivatives of basis functions as
well as input fields given in parametric coordinates. An input field that is
given in terms of physical coordinates only supports physical derivatives.

Note that the symbol :data:`dx` still includes the geometry Jacobian, and
therefore you should not multiply your expression with it if you want to
integrate over the parameter domain instead of the physical domain.  In this
case, you should multiply your expression with the attribute
:attr:`HSpace.GaussWeight` instead, which represents the weight for the Gauss
quadrature. Usually, the weight is automatically subsumed into :data:`dx`.


Supported functions
-------------------

The following functions and expressions are implemented in
:mod:`pyiga.vform` and have the same semantics as in the UFL language
(see the `UFL documentation`_):

:data:`dx`
:func:`Dx`
:func:`grad`
:func:`div`
:func:`curl`
:func:`as_vector`
:func:`as_matrix`
:func:`inner`
:func:`dot`
:func:`tr`
:func:`det`
:func:`inv`
:func:`cross`
:func:`outer`
:func:`abs`
:func:`sqrt`
:func:`exp`
:func:`log`
:func:`sin`
:func:`cos`
:func:`tan`

In addition, all expressions have the members :meth:`Expr.dx` for partial
derivatives (analogous to the global function :func:`Dx`),
:attr:`Expr.T` for transposing matrices,
:attr:`Expr.x`, :attr:`Expr.y`, :attr:`Expr.z` for accessing
vector components, and :meth:`Expr.dot()` which is analogous to the global
:func:`dot` function. Expressions can also be manipulated using the standard
arithmetic operators ``+,-,*,/`` as shown above.

.. _sec-compiling:

Compiling and assembling
------------------------

Once a vform has been defined, it has to be compiled into efficient code
and then invoked in order to assemble the desired matrix or vector.
Currently, there is only one backend in ``pyiga`` which is based on Cython --
the vform is translated into Cython code, compiled on the fly and loaded as a
dynamic module. The compiled modules are cached so that compiling a given vform for a
second time does not recompile the code.

Below is an example for assembling the Laplace variational form defined in
the section `A simple example`_::

    from pyiga import assemble, bspline, geometry

    # define the trial space
    kv = bspline.make_knots(3, 0.0, 1.0, 20)
    kvs = (kv, kv)   # 2D tensor product spline space

    # define the geometry map
    geo = geometry.quarter_annulus()   # NURBS quarter annulus

    A = assemble.assemble_vf(stiffness_vf(2), kvs, geo=geo, symmetric=True)

Any further input functions the assembler requires can be passed as further
keyword arguments in the call to :func:`.assemble_vf`. The function will
automatically detect whether the VForm has arity 1 or 2 and generate a vector
or a matrix correspondingly.

Manual compilation of the variational form
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Sometimes it may be necessary to directly work with the assembler class that
results from compiling a :class:`VForm`.  The functions used for compilation
are contained in the :mod:`pyiga.compile` module, and the resulting matrices
can be computed using the :func:`pyiga.assemble.assemble_entries` functions.

Using these functions, the Laplace variational form defined above can be
assembled as follows::

    from pyiga import compile, assemble, bspline, geometry

    # compile the vform into an assembler class
    Asm = compile.compile_vform(laplace_vf(2))

    # define the trial space
    kv = bspline.make_knots(3, 0.0, 1.0, 20)
    kvs = (kv, kv)   # 2D tensor product spline space

    # define the geometry map
    geo = geometry.quarter_annulus()   # NURBS quarter annulus

    A = assemble.assemble_entries(Asm(kvs, geo=geo), symmetric=True)

The geometry map is passed using ``geo=`` to the constructor of the
assembler class, and further input functions defined as described in
the section `Working with coefficient functions`_ can be passed in
the same way using their given name as the keyword.

The resulting object ``A`` is a sparse matrix in CSR format; different matrix
formats can be chosen by passing the ``format=`` keyword argument to
:func:`.assemble_entries`. The argument ``symmetric=True`` takes advantage of the
symmetry of the bilinear form in order to speed up the assembly.
