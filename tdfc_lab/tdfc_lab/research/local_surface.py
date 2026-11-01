"""Cached local-plane estimates for a known target cloud; no inter-agent data."""
import numpy as np


def local_surface(cloud, p, a, multiscale=False):
    key='_local_plane_multiscale_v1' if multiscale else '_local_plane_v1'
    if key not in cloud.meta:
        k=min(20,cloud.m)
        if k<6:
            return np.zeros_like(p),np.zeros_like(p),np.zeros(len(p))
        idx=cloud.tree.query(cloud.points,k=k)[1]
        patch=cloud.points[idx];center=patch.mean(axis=1);x=patch-center[:,None]
        values,vectors=np.linalg.eigh(np.einsum('nki,nkj->nij',x,x)/k)
        # Reject line-like supports and patches crossing sharp/multiple surfaces.
        planar=values[:,0]/np.maximum(values[:,1],1e-15)
        spread=values[:,1]/np.maximum(values[:,2],1e-15)
        confidence=np.clip((.08-planar)/.06,0,1)*np.clip((spread-.1)/.2,0,1)
        normal=vectors[:,:,0]
        if multiscale:
            # Prefer the largest reliable support; smaller supports only rescue
            # low-confidence patches. No mesh or evaluation points are queried.
            chosen=confidence>=.8
            for small in [12,8]:
                if small>=k:continue
                sp=patch[:,:small];sc=sp.mean(1);sx=sp-sc[:,None]
                sv,sn=np.linalg.eigh(np.einsum('nki,nkj->nij',sx,sx)/small)
                quality=np.clip((.08-sv[:,0]/np.maximum(sv[:,1],1e-15))/.06,0,1)*np.clip((sv[:,1]/np.maximum(sv[:,2],1e-15)-.1)/.2,0,1)
                take=(~chosen)&(quality>=.8)
                center[take]=sc[take];normal[take]=sn[take,:,0];confidence[take]=quality[take];chosen|=take
        cloud.meta[key]=(center,normal,confidence)
    center,normal,confidence=cloud.meta[key]
    distance,index=cloud.tree.query(p)
    q=center[index];n=normal[index]
    # Plane approximation used only near the target, smoothly faded outside.
    u=np.clip((.4*a-distance)/(.2*a),0,1);u=u*u*(3-2*u)
    return q,n,confidence[index]*u
