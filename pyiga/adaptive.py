import time
import math
import scipy
import numpy as np
import matplotlib as plt
from pyiga import assemble, adaptive, bspline, vform, geometry, vis, solvers, utils, topology
from sksparse.cholmod import cholesky
from pyiga import utils

### Error estimators
def PoissonEstimator(MP, uh, f=0., a=1., M=(0.,0.), neu_data={}, **kwargs):
    
    if isinstance(a,(np.ndarray,list,set)):
        assert len(a)==len(MP.mesh.domains)
        a={d:a_ for d,a_ in zip(MP.mesh.domains,a)}
    elif isinstance(a,(int,float)):
        a={d:a for d in MP.mesh.domains}

    if isinstance(f,(np.ndarray,list,set)):
        assert len(f)==len(MP.mesh.domains)
        f={d:f_ for d,f_ in zip(MP.mesh.domains,f)}
    elif isinstance(f,(int,float)):
        f={d:f for d in MP.mesh.domains}

        
    n = MP.mesh.numpatches
    indicator = np.zeros(n)
    uh_loc = MP.Basis@uh
    uh_per_patch = dict()
    
    #residual contribution, TODO vectorize
    t=time.time()
    for p, ((kvs, geo), _) in enumerate(MP.mesh.patches):
        h = np.linalg.norm([kv.meshsize_max() for kv in kvs])
        #h=np.linalg.norm([(b-a) for (a,b) in geo.bounding_box()])
        uh_per_patch[p] = uh_loc[np.arange(MP.N[p]) + MP.N_ofs[p]]   #cache Spline Function on patch p
        kvs0 = tuple([bspline.KnotVector(kv.mesh, 0) for kv in kvs])
        u_func = geometry.BSplineFunc(kvs, uh_per_patch[p])
        indicator[p] = h**2 * np.sum(assemble.assemble('(f + div(a*grad(uh)))**2 * v * dx', kvs=kvs0, geo=geo, a=a[MP.mesh.patch_domains[p]], f=f[MP.mesh.patch_domains[p]],uh=u_func,**kwargs))
    print('Residual contributions took ' + str(time.time()-t) + ' seconds.')
    #flux contribution
    t=time.time()
    for i,((p1,b1,_), (p2,b2,_), flip) in enumerate(MP.intfs):
        ((kvs1, geo1), _), ((kvs2, geo2), _) = MP.mesh.patches[p1], MP.mesh.patches[p2]
        bdspec1, bdspec2 = [assemble.int_to_bdspec(b1)], [assemble.int_to_bdspec(b2)]
        bkv1, bkv2 = assemble.boundary_kv(kvs1, bdspec1), assemble.boundary_kv(kvs2, bdspec2)
        geo = geo2.boundary(bdspec2)
        kv0 = tuple([bspline.KnotVector(kv.mesh, 0) for kv in bkv2])
        h = bkv2[0].meshsize_max()
        #h = np.linalg.norm([(b-a) for (a,b) in geo.bounding_box()])
        uh1_grad = geometry.BSplineFunc(kvs1, uh_per_patch[p1]).transformed_jacobian(geo1).boundary(bdspec1, flip=flip) #physical gradient of uh on patch 1 (flipped if needed)
        uh2_grad = geometry.BSplineFunc(kvs2, uh_per_patch[p2]).transformed_jacobian(geo2).boundary(bdspec2)            #physical gradient of uh on patch 2
        J = np.sum(assemble.assemble('(inner((a1 * uh1_grad) - (a2 * uh2_grad), n) )**2 * v * ds', kv0 ,geo=geo,a1=a[MP.mesh.patch_domains[p1]],a2=a[MP.mesh.patch_domains[p2]],uh1_grad=uh1_grad,uh2_grad=uh2_grad,M1=M[MP.mesh.patch_domains[p1]],M2=M[MP.mesh.patch_domains[p2]],**kwargs))
        indicator[p1] += 0.5 * h * J
        indicator[p2] += 0.5 * h * J
    for bd in neu_data:
        g = neu_data[bd]
        for (p,b) in MP.mesh.outer_boundaries[bd]:
            ((kvs, geo), _) = MP.mesh.patches[p]
            bdspec = [assemble.int_to_bdspec(b)]
            bkv = assemble.boundary_kv(kvs, bdspec)
            geo_b = geo.boundary(bdspec)
            uh_grad = geometry.BSplineFunc(kvs, uh_per_patch[p]).transformed_jacobian(geo).boundary(bdspec)
            J = np.sum(assemble.assemble('(inner(a * uh_grad, n) - g)**2 * v * ds', kv0 ,geo=geo_b, a=a[MP.mesh.patch_domains[p]], uh_grad=uh_grad, **kwargs))
            indicator[p] += h * J
    print('Jump contributions took ' + str(time.time()-t) + ' seconds.')
    return np.sqrt(indicator)

def ratio(kv,u,s=0):
    u=(1-u)*kv.support()[0]+u*kv.support()[1]
    if s==0:
        return np.clip(1-(kv.mesh[1:]-u)/(kv.mesh[1:]-kv.mesh[:-1]),a_min=0.,a_max=1.)
    else:
        return np.clip((kv.mesh[1:]-u)/(kv.mesh[1:]-kv.mesh[:-1]),a_min=0.,a_max=1.)
    
def PoissonEstimator2(MP, uh, f=0., a=1., M=(0.,0.), neu_data={}, **kwargs):
    if isinstance(a,(np.ndarray,list,set)):
        assert len(a)==len(MP.mesh.domains)
        a={d:a_ for d,a_ in zip(MP.mesh.domains,a)}
    elif isinstance(a,(int,float)):
        a={d:a for d in MP.mesh.domains}
    if isinstance(f,(np.ndarray,list,set)):
        assert len(f)==len(MP.mesh.domains)
        f={d:f_ for d,f_ in zip(MP.mesh.domains,f)}
    elif isinstance(f,(int,float)):
        f={d:f for d in MP.mesh.domains}
    n = MP.mesh.numpatches
    indicator = np.zeros((n,4))
    uh_loc = MP.Basis@uh
    uh_per_patch = dict()
    #residual contribution, TODO vectorize
    t=time.time()
    for p, ((kvs, geo), _) in enumerate(MP.mesh.patches):
        h = np.linalg.norm([(b-a)*kv.meshsize_max()/(kv.support()[1]-kv.support()[0]) for (a,b),kv in zip(geo.bounding_box(),kvs)])
        uh_per_patch[p] = uh_loc[np.arange(MP.N[p]) + MP.N_ofs[p]]   #cache Spline Function on patch p
        kvs0 = tuple([bspline.KnotVector(kv.mesh, 0) for kv in kvs])
        u_func = geometry.BSplineFunc(kvs, uh_per_patch[p])
        R=h**2*assemble.assemble('(f + div(a*grad(uh)))**2 * v * dx', kvs0, geo=geo, a=a[MP.mesh.patch_domains[p]],f=f[MP.mesh.patch_domains[p]],uh=u_func, M=M[MP.mesh.patch_domains[p]],**kwargs)
        for i,j in itertools.product(*2*(range(2),)):
            indicator[p,2*i+j] = np.sum(np.outer(ratio(kvs0[0],0.5,s=i),ratio(kvs0[1],0.5,s=j))*R)
    print('residual contributions took ' + str(time.time()-t) + ' seconds.')
    #flux contribution
    t=time.time()
    for i,((p1,b1,_), (p2,b2,_), flip) in enumerate(MP.intfs):
        ((kvs1, geo1), _), ((kvs2, geo2), _) = MP.mesh.patches[p1], MP.mesh.patches[p2]
        bdspec1, bdspec2 = [assemble.int_to_bdspec(b1)], [assemble.int_to_bdspec(b2)]
        bkv1, bkv2 = assemble.boundary_kv(kvs1, bdspec1), assemble.boundary_kv(kvs2, bdspec2)
        geo = geo2.boundary(bdspec2)
        kv0 = tuple([bspline.KnotVector(kv.mesh, 0) for kv in bkv2])
        h = np.sum(assemble.assemble('v * ds', kv0, geo=geo))*kv0[0].meshsize_max()/(kv0[0].support()[1]-kv0[0].support()[0])
        uh1_grad = geometry.BSplineFunc(kvs1, uh_per_patch[p1]).transformed_jacobian(geo1).boundary(bdspec1, flip=flip) #physical gradient of uh on patch 1 (flipped if needed)
        uh2_grad = geometry.BSplineFunc(kvs2, uh_per_patch[p2]).transformed_jacobian(geo2).boundary(bdspec2)            #physical gradient of uh on patch 2
        J = assemble.assemble('(inner((a1 * uh1_grad - a2 * uh2_grad + (M2 - M1)), n))**2 * v * ds', kv0 ,geo=geo,a1=a[MP.mesh.patch_domains[p1]],a2=a[MP.mesh.patch_domains[p2]],uh1_grad=uh1_grad,uh2_grad=uh2_grad,M1=M[MP.mesh.patch_domains[p1]],M2=M[MP.mesh.patch_domains[p2]],**kwargs)
        supp1, supp2 = geo1.boundary(bdspec1).support(), geo2.boundary(bdspec2).support()
        if b1==0: s10,s11=0,1
        if b1==1: s10,s11=2,3
        if b1==2: s10,s11=0,2
        if b1==3: s10,s11=1,3
        if min(supp2)>0.5*sum(supp1): s1=s11
        else: s1=s12
        indicator[p1,s1 ] += 0.5 * h * np.sum(J)
        if b2==0: s20,s21=0,1
        if b2==1: s20,s21=2,3
        if b2==2: s20,s21=0,2
        if b2==3: s20,s21=1,3
        indicator[p2,s20] += 0.5 * h * np.sum(ratio(kv0,0.5,s=0)*J)
        indicator[p2,s21] += 0.5 * h * np.sum(ratio(kv0,0.5,s=1)*J)
    for bd in neu_data:
        g = neu_data[bd]
        for (p,b) in MP.mesh.outer_boundaries[bd]:
            ((kvs, geo), _) = MP.mesh.patches[p]
            bdspec = [assemble.int_to_bdspec(b)]
            bkv = assemble.boundary_kv(kvs, bdspec)
            geo_b = geo.boundary(bdspec)
            uh_grad = geometry.BSplineFunc(kvs, uh_per_patch[p]).transformed_jacobian(geo).boundary(bdspec)
            J = np.sum(assemble.assemble('(inner(a * uh_grad, n) - g)**2 * v * ds', kv0 ,geo=geo_b, a=a[MP.mesh.patch_domains[p]], uh_grad=uh_grad, **kwargs))
            indicator[p] += h * J
    print('jump contributions took ' + str(time.time()-t) + ' seconds.')
    return np.sqrt(indicator)

def doerfler_marking(errors, theta=0.8):
    """Given a list of errors, return a minimal list of indices such that the indexed
    errors have norm of at least theta * norm(errors)."""
    ix = np.argsort(errors)
    total = np.linalg.norm(errors)
    running = []
    marked = []
    for i in reversed(ix):
        running.append(errors[i])
        marked.append(i)
        if np.linalg.norm(running) >= theta * total:
            break
    return marked
