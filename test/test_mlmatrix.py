from pyiga.mlmatrix import *

def test_tofrom_seq():
    for i in range(3*4*5):
        assert to_seq(from_seq(i, (3,4,5)), (3,4,5)) == i

def test_tofrom_multilevel():
    bs = np.array(((3,3), (4,4), (5,5)))  # block sizes for each level
    for i in range(3*3 + 4*4 + 5*5):
        for j in range(3*3 + 4*4 + 5*5):
            assert reindex_from_multilevel(reindex_to_multilevel(i, j, bs), bs) == (i,j)

def test_banded_sparsity():
    n = 10
    bw = 2

    X = np.zeros((n,n), dtype=int)
    for i in range(n):
        for j in range(n):
            if abs(i-j) <= bw:
                X[i,j] = 1
    assert np.array_equal(np.flatnonzero(X),
                          compute_banded_sparsity(n, bw))
