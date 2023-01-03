import itertools as it

def face_indices(n, m, zipped=True):
    """Returns all m-dimensional boundary manifolds of an n-dimensional hypercube.
    
    The manifolds are represented by a list of pairs ('ax', 'side') where 'ax' is an integer 
    describing which axis we set to 'side' which is 0 or 1.
    
    Optionally a parameter 'zipped' can be passed, which is by default 'True'. If 'zipped' is set to False,
    the manifolds are given as 2 tuples ('axis', 'sides') which saves the 'axis' and corresponding 'sides' information seperately.
    """
    S=[]
    for comb in it.combinations(range(n),n-m):
        for i in it.product(*(n-m)*((0,1),)):
            #print(list(zip(comb,i)))
            if zipped:
                S.append(tuple(zip(comb,i)))
            else:
                S.append((comb, i))
    return S