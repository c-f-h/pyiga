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

def test_basisderivs():
    # scalar basis functions
    vf = VForm(3, arity=1)
    u = vf.basisfuns()
    assert grad(u).shape == (3,)

    # vector basis functions
    vf = VForm(3, arity=1)
    u = vf.basisfuns(components=(3,))
    assert div(u).shape == ()
    assert curl(u).shape == (3,)
