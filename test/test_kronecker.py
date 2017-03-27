from pyiga.kronecker import *

def test_kronecker_2d():
    X = np.random.rand(8,8)
    Y = np.random.rand(8,8)
    XY = np.kron(X, Y)

    x = np.random.rand(8**2)
    y1 = apply_kronecker((X,Y), x)
    y2 = apply_tprod((X,Y), x.reshape(8,8))
    assert abs(XY.dot(x) - y1).max() < 1e-10
    assert abs(XY.dot(x) - y2.ravel()).max() < 1e-10

    x = np.random.rand(8**2, 1)
    assert abs(XY.dot(x) - apply_kronecker((X,Y), x)).max() < 1e-10

    x = np.random.rand(8**2, 7)
    assert abs(XY.dot(x) - apply_kronecker((X,Y), x)).max() < 1e-10

def test_kronecker_3d():
    X = np.random.rand(8,8)
    Y = np.random.rand(8,8)
    Z = np.random.rand(8,8)
    XYZ = np.kron(np.kron(X, Y), Z)

    x = np.random.rand(8**3)
    y1 = apply_kronecker((X,Y,Z), x)
    y2 = apply_tprod((X,Y,Z), x.reshape(8,8,8))
    assert abs(XYZ.dot(x) - y1).max() < 1e-10
    assert abs(XYZ.dot(x) - y2.ravel()).max() < 1e-10

    x = np.random.rand(8**3, 1)
    assert abs(XYZ.dot(x) - apply_kronecker((X,Y,Z), x)).max() < 1e-10

    x = np.random.rand(8**3, 7)
    assert abs(XYZ.dot(x) - apply_kronecker((X,Y,Z), x)).max() < 1e-10
