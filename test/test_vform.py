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
    exprs_equal(f**1, f)
    exprs_equal(f**2, f*f)
    exprs_equal(f**as_expr(3), f*f*f)
    exprs_equal(f**0, as_expr(1))
    exprs_equal(f**-1, 1.0 / f)
    exprs_equal(f**-2, 1.0 / (f*f))

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
    G = as_matrix(grad(vf.Geo, parametric=True))
    assert G.shape == (2,2)
    G = as_matrix(2 * grad(vf.Geo, parametric=True))
    assert G.shape == (2,2)

def exprs_equal(expr1, expr2, simplify=False):
    if simplify:
        from pyiga.vform import _to_literal_vec_mat
        def simpl(expr):
            expr = transform_expr(expr, _to_literal_vec_mat)
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
    exprs_equal(vf.Geo.dx(1, parametric=True)[0], vf.Geo[0].dx(1, parametric=True))
    G = vf.let('G', vf.Geo)
    exprs_equal(G.dx(0, parametric=True)[1], G[1].dx(0, parametric=True))

def test_vectorexpr():
    vf = VForm(3)
    u, v = vf.basisfuns(components=(3,3))
    A = vf.input('A', shape=(3,3))
    assert inner(u, v).shape == ()
    assert cross(u, v).shape == (3,)
    assert outer(u, v).shape == (3, 3)
    assert A.dot(u).shape == (3,)
    x = (1, 2, 3)
    assert inner(x, v).shape == ()
    assert cross(x, v).shape == (3,)
    assert outer(x, v).shape == (3, 3)
    assert A.dot(x).shape == (3,)

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
    # expressions with parametric derivatives
    assert grad(f, parametric=True).shape == (3,)
    assert grad(g, parametric=True).shape == (3,3)
    assert grad(f, dims=(1,2), parametric=True).shape == (2,)
    exprs_equal(grad(f, dims=(1,2), parametric=True)[0], Dx(f, 1, parametric=True))
    assert grad(g, dims=(1,2), parametric=True).shape == (3,2)
    exprs_equal(grad(g, dims=(1,2), parametric=True)[1,0], Dx(g[1], 1, parametric=True))
    exprs_equal(grad(g, parametric=True)[1, :], grad(g[1], parametric=True))
    # expressions with physical derivatives
    assert grad(f).shape == (3,)
    assert grad(g).shape == (3,3)
    assert grad(f, dims=(1,2)).shape == (2,)
    exprs_equal(grad(f, dims=(1,2))[0], Dx(f, 1))
    assert grad(g, dims=(1,2)).shape == (3,2)
    exprs_equal(grad(g, dims=(1,2))[1,0], Dx(g[1], 1))
    exprs_equal(grad(g)[1, :], grad(g[1]))

def test_input_physical():
    vf = VForm(3, arity=1)
    f = vf.input('f', physical=True)
    g = vf.input('g', shape=(3,), physical=True)
    G = vf.input('G', shape=(3,3), physical=True)
    assert f.shape == ()
    assert g.shape == (3,)
    assert G.shape == (3,3)
    # expressions with physical derivatives
    assert grad(f).shape == (3,)
    exprs_equal(grad(f)[1:], grad(f, dims=[1,2]))
    assert grad(g).shape == (3,3)
    assert grad(f, dims=(1,2)).shape == (2,)
    exprs_equal(grad(f, dims=(1,2))[0], Dx(f, 1))
    assert grad(g, dims=(1,2)).shape == (3,2)
    exprs_equal(grad(g, dims=(1,2))[1,0], Dx(g[1], 1))
    exprs_equal(grad(g)[1, :], grad(g[1]))

def test_symderiv():
    vf = VForm(3, arity=1)
    u = vf.basisfuns()
    f = vf.input('f')
    G = vf.input('G', shape=(3,))
    exprs_equal(grad(2 * f, parametric=True), 2 * grad(f, parametric=True), simplify=True)
    exprs_equal(div(G - 3, parametric=True), div(G, parametric=True), simplify=True)
    exprs_equal((f * u).dx(0, parametric=True), f.dx(0, parametric=True)*u + f*u.dx(0, parametric=True))
    exprs_equal((1 / f).dx(1, parametric=True), -f.dx(1, parametric=True) / (f*f), simplify=True)
    exprs_equal(curl(2 + grad(u)), curl(grad(u)), simplify=True)

def test_hess():
    vf = VForm(3, arity=1)
    f = vf.input('f')
    v = vf.basisfuns()
    Hp = hess(f, parametric=True)
    assert Hp.shape == (3,3)
    exprs_equal(Hp[0,1], Dx(Dx(f, 0, parametric=True), 1, parametric=True))
    exprs_equal(Hp[2,0], Dx(Dx(f, 0, parametric=True), 2, parametric=True))
    H = hess(f)
    assert H.shape == (3,3)
    exprs_equal(H[1,1], Dx(Dx(f, 1), 1))
    exprs_equal(H[1,2], Dx(Dx(f, 2), 1))
    # test finalize
    vf.add(inner(H, H) * v * dx)
    vf.finalize()

def test_tostring():
    vf = VForm(2)
    u, v = vf.basisfuns()
    f = vf.input('f')
    g = vf.input('g', shape=(2,))
    B = vf.input('B', shape=(2,2))
    assert str(f) == 'f_a'  # implementation detail of current generators
    assert str(-f) == '-f_a'
    assert str(cos(f)) == 'cos(f_a)'
    assert str(g[1]) == 'g_a[1]'
    tmp = f + 1
    assert str(tmp) == '+(f_a, 1.0)'
    tmp1 = vf.let('tmp1', tmp)
    assert str(tmp1) == 'tmp1'
    vec = as_vector((2, 3))
    assert str(vec) == '(2.0, 3.0)'
    mat = as_matrix(((1,2),(3,4)))
    assert str(mat) == '((1.0, 2.0),\n (3.0, 4.0))'
    assert str(u) == 'u'
    assert str(Dx(u, 1)) == 'u_01(phys)'
    assert str(B[1,0]) == 'B_a[1,0]'
    assert str(B.dot(g)) == 'dot(((B_a[0,0], B_a[0,1]),\n (B_a[1,0], B_a[1,1])), (g_a[0], g_a[1]))'
    assert str(B.dot(B)) == 'dot(((B_a[0,0], B_a[0,1]),\n (B_a[1,0], B_a[1,1])), ((B_a[0,0], B_a[0,1]),\n (B_a[1,0], B_a[1,1])))'

    tree_print(tmp)

def test_surface():
    vf = VForm(1, geo_dim=2)
    assert (vf.dim, vf.geo_dim) == (1, 2)
    assert vf.normal.shape == (2,)
    assert vf.SW.shape == ()        # check surface weight

    vf = VForm(2, geo_dim=3)
    assert (vf.dim, vf.geo_dim) == (2, 3)
    assert vf.normal.shape == (3,)
    assert vf.SW.shape == ()        # check surface weight

def test_parse():
    from pyiga import bspline, geometry
    kvs = 2 * (bspline.make_knots(2, 0.0, 1.0, 5),)

    vf = parse_vf('u * v * dx', kvs, bfuns=[('u', 1), ('v', 1)])
    assert vf.hash() == mass_vf(2).hash()

    f = bspline.BSplineFunc(kvs, np.ones(bspline.numdofs(kvs)))
    vf = parse_vf('f * v * dx', kvs, {'f': f})
    assert vf.hash() == L2functional_vf(2, physical=False).hash()

    f = lambda x, y: 1.0
    vf = parse_vf('f * v * dx', kvs, {'f': f})
    assert vf.hash() == L2functional_vf(2, physical=True).hash()

    vf = parse_vf('div(u) * div(v) * dx', kvs, bfuns=[('u', 2), ('v', 2)])
    assert vf.hash() == divdiv_vf(2).hash()

    # some other features
    vf = parse_vf('f * v * ds', kvs[:1], {'f': geometry.circular_arc(1.4)[0]})
