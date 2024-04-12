import time
import math
import scipy
import numpy as np
import matplotlib as plt
from pyiga import assemble, adaptive, bspline, vform, geometry, vis, solvers, utils, topology
from sksparse.cholmod import cholesky
from pyiga import utils

################################################################################
# Error Estimation
################################################################################

def mp_resPois(MP, uh, f=0., a=1., M=(0.,0.), divMaT =0., neu_data={}, **kwargs):
    if isinstance(a,(int,float)):
        a={d:a for d in MP.mesh.domains}
    if isinstance(f,(int,float)):
        f={d:f for d in MP.mesh.domains}
    n = MP.mesh.numpatches
    indicator = np.zeros(n)
    uh_loc = MP.Basis@uh
    uh_per_patch = dict()
    
    #residual contribution
    t=time.time()
    
    #slightly faster
    # kvs, geos = MP.mesh.kvs, MP.mesh.geos
    # h_V = np.array([[kv.meshsize_max()*(b-a) for kv,(a,b) in zip(kvs_,geo.bounding_box(full=True))] for kvs_,geo in zip(kvs,geos)])
    # h_V = np.linalg.norm(h_V,axis=1)
    # kvs0 = [tuple([bspline.KnotVector(kv.mesh, 0) for kv in kvs_]) for kvs_ in kvs]
    # R = np.array([assemble.assemble('(f + div(a*grad(uh))) * v * dx', kvs=kv0 , geo=geo , a=a[MP.mesh.patch_domains[p]], f=f[MP.mesh.patch_domains[p]] ,uh=geometry.BSplineFunc(kv, uh_loc[MP.N_ofs[p]:MP.N_ofs[p+1]])).ravel() for p, (kv0, kv, geo) in enumerate(zip(kvs0,kvs,geos))])
    # indicator = h_V**2 * R.sum(axis=1)
    
    for p, ((kvs, geo), _) in enumerate(MP.mesh.patches):
        h = np.linalg.norm([kv.meshsize_max()*(b-a) for kv,(a,b) in zip(kvs,geo.bounding_box(full=True))])
        #h=np.linalg.norm([(b-a) for (a,b) in geo.bounding_box()])
        uh_per_patch[p] = uh_loc[np.arange(MP.N[p]) + MP.N_ofs[p]]   #cache Spline Function on patch p
        kvs0 = tuple([bspline.KnotVector(kv.mesh, 0) for kv in kvs])
        u_func = geometry.BSplineFunc(kvs, uh_per_patch[p])
        indicator[p] = h**2 * np.sum(assemble.assemble('((f + div(a*grad(uh)))**2 * v) * dx', kvs=kvs0, geo=geo, a=a[MP.mesh.patch_domains[p]], f=f[MP.mesh.patch_domains[p]],uh=u_func,**kwargs))
    print('Residual contributions took ' + str(time.time()-t) + ' seconds.')
    
    #flux contribution
    t=time.time()
    for i,((p1,b1,_), (p2,b2,_), flip) in enumerate(MP.intfs):
        ((kvs1, geo1), _), ((kvs2, geo2), _) = MP.mesh.patches[p1], MP.mesh.patches[p2]
        bdspec1, bdspec2 = [assemble.int_to_bdspec(b1)], [assemble.int_to_bdspec(b2)]
        bkv1, bkv2 = assemble.boundary_kv(kvs1, bdspec1), assemble.boundary_kv(kvs2, bdspec2)
        geo = geo2.boundary(bdspec2)
        kv0 = tuple([bspline.KnotVector(kv.mesh, 0) for kv in bkv2])
        h = bkv2[0].meshsize_max()*np.linalg.norm([b-a for a,b in geo.bounding_box(full=True)])
        #h = np.linalg.norm([(b-a) for (a,b) in geo.bounding_box()])
        uh1_grad = geometry.BSplineFunc(kvs1, uh_loc[MP.N_ofs[p1]:MP.N_ofs[p1+1]]).transformed_jacobian(geo1).boundary(bdspec1, flip=flip) #physical gradient of uh on patch 1 (flipped if needed)
        uh2_grad = geometry.BSplineFunc(kvs2, uh_loc[MP.N_ofs[p2]:MP.N_ofs[p2+1]]).transformed_jacobian(geo2).boundary(bdspec2)            #physical gradient of uh on patch 2
        J = np.sum(assemble.assemble('((inner((a1 * uh1_grad + Ma1) - (a2 * uh2_grad + Ma2), n) )**2 * v ) * ds', kv0 ,geo=geo,a1=a[MP.mesh.patch_domains[p1]],a2=a[MP.mesh.patch_domains[p2]],uh1_grad=uh1_grad,uh2_grad=uh2_grad,Ma1=M[MP.mesh.patch_domains[p1]],Ma2=M[MP.mesh.patch_domains[p2]],**kwargs))
        indicator[p1] += 0.5 * h * J
        indicator[p2] += 0.5 * h * J
        
    #Neumann flux
    for bd in neu_data:
        g = neu_data[bd]
        for (p,b) in MP.mesh.outer_boundaries[bd]:
            ((kvs, geo), _) = MP.mesh.patches[p]
            bdspec = [assemble.int_to_bdspec(b)]
            bkv = assemble.boundary_kv(kvs, bdspec)
            kv0 = tuple([bspline.KnotVector(kv.mesh, 0) for kv in bkv])
            geo_b = geo.boundary(bdspec)
            uh_grad = geometry.BSplineFunc(kvs, uh_per_patch[p]).transformed_jacobian(geo).boundary(bdspec)
            J = np.sum(assemble.assemble('((inner(a * uh_grad + Ma, n) - g)**2 * v ) * ds', kv0 ,geo=geo_b,Ma=M[MP.mesh.patch_domains[p]], a=a[MP.mesh.patch_domains[p]],g=g, uh_grad=uh_grad, **kwargs))
            indicator[p] += h * J
            
    print('Jump contributions took ' + str(time.time()-t) + ' seconds.')
    return np.sqrt(indicator)

def ratio(kv,u,s=0):
    u=(1-u)*kv.support()[0]+u*kv.support()[1]
    if s==0:
        return np.clip(1-(kv.mesh[1:]-u)/(kv.mesh[1:]-kv.mesh[:-1]),a_min=0.,a_max=1.)
    else:
        return np.clip((kv.mesh[1:]-u)/(kv.mesh[1:]-kv.mesh[:-1]),a_min=0.,a_max=1.)

################################################################################
# Marking
################################################################################

def doerfler_mark(x, theta=0.8, TOL=0.01):
    """Given an array of x, return a minimal array of indices such that the indexed
    values of x have norm of at least theta * norm(errors). Requires sorting the array x.
    Indices of entries that are 100*TOL percentage off from the breakpoint entry are also added to the output"""
    idx = np.argsort(x)
    n=len(idx)
    total = x@x
    S=0
    for i in reversed(range(n)):
        S+= x[idx[i]]**2
        if (S > theta * total):
            k=i
            while (abs(x[idx[i]]-x[idx[k]])/x[idx[i]] < TOL) and k>0:       #we go on adding entries that are just 100*TOL% off from the breakpoint entry.
                k-=1
            break
    return idx[k:]

def quick_mark(x, idx = None, l=None, u=None , v=None, theta=0.8):
    """Given an array of x, return a minimal array of indices such that the indexed
    values of x have norm of at least theta * norm(errors)**2. Does not require sorting the array x.
    TODO: add checks for when values are equal in the array, see Praetorius 2019 paper."""
    if idx is None: idx=np.arange(len(x))
    if l is None: l=0
    if u is None: u=len(x)-1
    if v is None: v=theta*x@x
        
    p = l+(u-l)//2                                                           #pivot for partition is chosen as the median
    idx[l:(u+1)] = idx[l:(u+1)][np.argpartition(-x[idx[l:(u+1)]],p-l)]       #partition of subarray from l to u
    sigma = x[idx[l:p]]@x[idx[l:p]]
    if sigma > v:                                                            #if the norm of the larger entries exceeds the total norm we didn't find the minimal set of entries yet.
        return quick_mark(x, idx, l, p-1, v, theta=theta)
    elif sigma + x[idx[p]]**2 > v:                                           #if adding the p-th value (the next biggest entry we can add) suddenly satisfies the condition we are done.
        return idx[:(p+1)]# idx[:(p + ceil((v-sigma)/x[idx[p]]))             
    else:                                                                    #we haven't reached the desired norm so we have to look further.
        return quick_mark(x, idx, p + 1,u,v-sigma-x[idx[p]]**2,theta=theta)