"""Fixed metric 1D target curves and an explicitly separate curve controller.

No area-based N scaling or analytic surface normals are used. The public curve
prior provides positions/tangents; only visible robot states enter interactions.
These are point-mass preflight examples, not downwash/flight-dynamics models.
"""
import numpy as np
from scipy.spatial import cKDTree
from .geometry import PointCloud
from .energy import bump_kernel_dr


def _parametric(shape,t):
    if shape.startswith('curve_demo_'):
        from .curve_catalog import catalog_parametric
        return catalog_parametric(shape,t)
    if shape=='curve_helix':
        angle=2.5*np.pi*t
        return np.column_stack([.65*np.cos(angle),.65*np.sin(angle),.6+1.2*t])
    if shape=='curve_ring':
        angle=2*np.pi*t
        return np.column_stack([.9*np.cos(angle),.9*np.sin(angle),np.full_like(t,1.2)])
    if shape=='curve_wave':
        return np.column_stack([4*(t-.5),.45*np.sin(2*np.pi*t),1.2+.35*np.sin(4*np.pi*t)])
    raise ValueError(shape)


def curve_geometry(shape,control_points=4096,eval_points=32768):
    dense=_parametric(shape,np.linspace(0,1,65537))
    arc=np.r_[0,np.cumsum(np.linalg.norm(np.diff(dense,axis=0),axis=1))];length=arc[-1]
    def sample(count,offset):
        # Equal arclength quadrature: coverage is length-weighted, not parameter-weighted.
        distance=(np.arange(count)+offset)*length/count
        p=np.column_stack([np.interp(distance,arc,dense[:,axis]) for axis in range(3)])
        cloud=PointCloud.from_points(p,area=0.,name=shape)
        parameter=np.interp(distance,arc,np.linspace(0,1,len(dense)))
        eps=1e-6;derivative=_parametric(shape,parameter+eps)-_parametric(shape,parameter-eps)
        tangent=derivative/np.linalg.norm(derivative,axis=1)[:,None]
        cloud.meta.update(dimension=1,length=float(length),tangents=tangent,
                          area_unused=True,fixed_metric_geometry=True)
        if shape.startswith('curve_demo_'):
            from .curve_catalog import parse,CLOSED
            kind,n=parse(shape)
            cloud.meta.update(closed_curve=kind in CLOSED,catalog_kind=kind,catalog_agents=n,
                              length_rule='0.7N metres, fixed before screening')
        return cloud
    return sample(control_points,0.),sample(eval_points,.5)


def curve_initial_state(n,kind,seed,d_safe):
    rng=np.random.default_rng(seed);spacing=1.4*d_safe
    x,y,z=np.meshgrid(np.arange(-2,3)*spacing,np.arange(-2,3)*spacing,
                      np.arange(-1,2)*spacing,indexing='ij')
    sites=np.column_stack([x.ravel(),y.ravel(),z.ravel()])
    if kind=='surround':candidates=sites
    elif kind=='one_side':candidates=sites[sites[:,0]==-2*spacing]
    elif kind=='cluster':
        order=np.argsort(np.linalg.norm(sites-np.array([-spacing,0,0]),axis=1),kind='stable')
        candidates=sites[order[:max(n,12)]]
    elif kind=='multi_cluster':candidates=sites[np.abs(sites[:,0])==2*spacing]
    else:raise ValueError(kind)
    if kind=='multi_cluster':
        left=candidates[candidates[:,0]<0];right=candidates[candidates[:,0]>0]
        p=np.concatenate([left[rng.choice(len(left),n//2,replace=False)],right[rng.choice(len(right),n-n//2,replace=False)]])
    else:p=candidates[rng.choice(len(candidates),n,replace=False)].copy()
    p+=rng.uniform(-.025*d_safe,.025*d_safe,p.shape);p[:,2]+=1.2
    if cKDTree(p).query(p,k=2)[0][:,1].min()<1.2*d_safe:raise ValueError('unsafe curve initialization')
    return p,np.zeros_like(p)


def curve_acceleration(p,v,cloud,prm,limit,endpoint_cone=False):
    d,indices=cloud.tree.query(p);q=cloud.points[indices];tangent=cloud.meta['tangents'][indices]
    delta=p[:,None]-p[None];r=np.linalg.norm(delta,axis=2)
    visible=(r>1e-12)&(r<prm.r_vis_abs)
    unit=delta/np.maximum(r[:,:,None],1e-12)
    soft=np.where(visible,prm.k_p*np.maximum(prm.r_c-r,0),0)
    density=np.where(visible,-prm.gamma*bump_kernel_dr(r,prm.h),0)
    pair=((soft+density)[:,:,None]*unit).sum(1)
    tangential=np.sum(pair*tangent,axis=1)[:,None]*tangent
    endpoint=np.zeros(len(p),dtype=bool)
    if endpoint_cone and not cloud.meta.get('closed_curve',cloud.name=='curve_ring'):
        first=indices==0;last=indices==cloud.m-1;endpoint=first|last
        outward=(first&(np.sum(tangential*tangent,axis=1)<0))|(last&(np.sum(tangential*tangent,axis=1)>0))
        # Tangent cone of the closed curve interval: never generate outward
        # redistribution at an endpoint. This is not an inertial invariance proof.
        tangential[outward]=0.
    vt=np.sum(v*tangent,axis=1)[:,None]*tangent
    # At open-curve endpoints the full nearest-point attraction also confines
    # motion along the endpoint tangent; away from endpoints this is normal.
    acc=-prm.k_s*(p-q)+tangential-prm.c*vt-max(prm.c,2*np.sqrt(prm.k_s))*(v-vt)
    if endpoint_cone:
        acc[endpoint]-=(max(prm.c,2*np.sqrt(prm.k_s))-prm.c)*vt[endpoint]
    acc*=np.minimum(1,limit/np.maximum(np.linalg.norm(acc,axis=1),1e-30))[:,None]
    normal=(p-q)/np.maximum(d[:,None],1e-12)
    return acc,normal
