from pyiga.vform import *

def test_arithmetic():
    vf = VForm(2)
    u, v = vf.basisfuns()
    f, g = vf.input('f'), vf.input('g')
    assert (f + g).shape == ()
    assert (f - g).shape == ()
    assert (f * g).shape == ()
    assert (f / g).shape == ()
    assert (f + 2).shape == ()
    assert (f - 2).shape == ()
    assert (f * 2).shape == ()
    assert (f / 2).shape == ()
    assert (3 + g).shape == ()
    assert (3 - g).shape == ()
    assert (3 * g).shape == ()
    assert (3 / g).shape == ()
    assert (3 * grad(u)).shape == (2,)
    assert (grad(v) / 3).shape == (2,)

def test_asvector():
    vf = VForm(2)
    G = as_vector([1,2,3])
    assert G.shape == (3,)
    G = as_vector(vf.Geo)
    assert G.shape == (2,)
    G = as_vector(2 * vf.Geo)
    assert G.shape == (2,)

def test_asmatrix():
    vf = VForm(2)
    G = as_matrix([[1,2,3],[4,5,6]])
    assert G.shape == (2,3)
    G = as_matrix(grad(vf.Geo))
    assert G.shape == (2,2)
    G = as_matrix(2 * grad(vf.Geo))
    assert G.shape == (2,2)

def exprs_equal(expr1, expr2, simplify=False):
    if simplify:
        from pyiga.vform import _literalize_helper
        def simpl(expr):
            expr = transform_expr(expr, _literalize_helper)
            expr = transform_expr(expr, lambda e: e.fold_constants())
            return expr
        expr1 = simpl(expr1)
        expr2 = simpl(expr2)

    h1, h2 = exprhash(expr1), exprhash(expr2)
    if h1 != h2:
        print('Expression 1:')
        tree_print(expr1)
        print('Expression 2:')
        tree_print(expr2)
    assert h1 == h2

def test_dx():
    vf = VForm(2)
    u, v = vf.basisfuns()
    exprs_equal(u.dx(0).dx(1), u.dx(1).dx(0))
    exprs_equal(vf.Geo.dx(1)[0], vf.Geo[0].dx(1))
    G = vf.let('G', vf.Geo)
    exprs_equal(G.dx(0)[1], vf.Geo[1].dx(0))

def test_vectorexpr():
    vf = VForm(3)
    u, v = vf.basisfuns(components=(3,3))
    A = vf.input('A', shape=(3,3))
    assert inner(u, v).shape == ()
    assert cross(u, v).shape == (3,)
    assert outer(u, v).shape == (3, 3)
    assert A.dot(u).shape == (3,)

def test_basisderivs():
    # scalar basis functions
    vf = VForm(3, arity=1)
    u = vf.basisfuns()
    assert grad(u).shape == (3,)

    # vector basis functions
    vf = VForm(3, arity=1)
    u = vf.basisfuns(components=(3,))
    assert grad(u).shape == (3,3)
    assert grad(u, dims=(1,2)).shape == (3,2)
    assert div(u).shape == ()
    assert curl(u).shape == (3,)

def test_input():
    vf = VForm(3, arity=1)
    f = vf.input('f')
    g = vf.input('g', shape=(3,))
    G = vf.input('G', shape=(3,3))
    assert f.shape == ()
    assert g.shape == (3,)
    assert G.shape == (3,3)
    assert grad(f).shape == (3,)
    assert grad(g).shape == (3,3)
    assert grad(f, dims=(1,2)).shape == (2,)
    exprs_equal(grad(f, dims=(1,2))[0], Dx(f, 1))
    assert grad(g, dims=(1,2)).shape == (3,2)
    exprs_equal(grad(g, dims=(1,2))[1,0], Dx(g[1], 1))

def test_symderiv():
    vf = VForm(3, arity=1)
    u = vf.basisfuns()
    f = vf.input('f')
    G = vf.input('G', shape=(3,))
    exprs_equal(grad(2 * f), 2 * grad(f), simplify=True)
    exprs_equal(div(G - 3), div(G), simplify=True)
    exprs_equal((f * u).dx(0), f.dx(0)*u + f*u.dx(0))
    exprs_equal((1 / f).dx(1), -f.dx(1) / (f*f), simplify=True)
    exprs_equal(curl(2 + grad(u)), curl(grad(u)), simplify=True)
