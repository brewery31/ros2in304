"""Independent implementation of Yue et al., arXiv:2509.26518v2, Sec. II.

This is not author code. The target is explicitly voxelized as a thin shell;
physical map size is supplied by the common task rather than rescaled by the
paper's occupied-volume rule. No memory-reduction claim is made here.
"""
from __future__ import annotations
import numpy as np
from scipy.spatial import cKDTree


class TreeShape:
    def __init__(self, points, depth=6, solid_sphere=False):
        self.depth=depth;self.resolution=2**depth
        p=np.asarray(points,float);center=(p.min(0)+p.max(0))/2
        self.extent=float(np.ptp(p,axis=0).max()*1.08)
        self.origin=center-self.extent/2;self.pixel=self.extent/self.resolution
        # One occupied voxel for each target sample; no filling of target interior.
        index=np.floor((p-self.origin)/self.pixel).astype(int)
        index=np.clip(index,0,self.resolution-1)
        grid=np.zeros((self.resolution,)*3,bool);grid[tuple(index.T)]=True
        if solid_sphere:
            # Native volumetric reference case, kept separate from shell tests.
            center_sphere=np.linalg.lstsq(2*(p[1:]-p[0]),(p[1:]**2).sum(1)-(p[0]**2).sum(),rcond=None)[0]
            radius=float(np.linalg.norm(p-center_sphere,axis=1).mean())
            coords=self.origin+(np.indices(grid.shape).reshape(3,-1).T+.5)*self.pixel
            grid=(np.linalg.norm(coords-center_sphere,axis=1)<=radius).reshape(grid.shape)
        self.grid=grid;self.nodes=[]
        def build(lo,span,parent):
            node=len(self.nodes);self.nodes.append(None)
            region=grid[lo[0]:lo[0]+span,lo[1]:lo[1]+span,lo[2]:lo[2]+span]
            beta=float(region.mean());children=[]
            if span>1 and 0<beta<1:
                half=span//2
                for b in range(8):
                    offset=np.array([(b>>2)&1,(b>>1)&1,b&1])*half
                    children.append(build(lo+offset,half,node))
            self.nodes[node]=dict(lo=lo.copy(),span=span,parent=parent,beta=beta,
                children=children,center=self.origin+(lo+span/2)*self.pixel)
            return node
        build(np.zeros(3,int),self.resolution,-1)
        self.all_centers=np.array([n['center'] for n in self.nodes])
        self.all_volumes=np.array([(n['span']*self.pixel)**3 for n in self.nodes])
        self.black=np.array([i for i,n in enumerate(self.nodes) if not n['children'] and n['beta']==1],int)
        self.centers=np.array([self.nodes[i]['center'] for i in self.black])
        self.sizes=np.array([self.nodes[i]['span']*self.pixel for i in self.black])
        self.black_tree=cKDTree(self.centers)
        self.max_half_diagonal=float(self.sizes.max()*np.sqrt(3)/2)
        # Finest occupied virtual-node centers. Equivalent to generating these
        # from black leaves on demand, but retained for faster repeated queries.
        # This optimization means our storage is not the paper's compact budget.
        self.virtual=self.origin+(np.argwhere(grid)+.5)*self.pixel
        self.virtual_tree=cKDTree(self.virtual)
        self.quantization_bound=np.sqrt(3)*self.pixel

    def contains(self,p):
        ind=np.floor((np.asarray(p)-self.origin)/self.pixel).astype(int)
        inside=np.all((ind>=0)&(ind<self.resolution),axis=1)
        result=np.zeros(len(ind),bool);valid=ind[inside]
        result[inside]=self.grid[tuple(valid.T)]
        return result

    def locate(self,p):
        inside=np.all((p>=self.origin)&(p<self.origin+self.extent))
        if not inside:return None
        i=0
        while self.nodes[i]['children']:
            n=self.nodes[i];bits=p>=n['center'];b=int(bits@np.array([4,2,1]))
            i=n['children'][b]
        return i

    def entering_leaf(self,p):
        leaf=self.locate(p)
        if leaf is None:
            i=0
            while self.nodes[i]['children']:
                children=[j for j in self.nodes[i]['children'] if self.nodes[j]['beta']>0]
                i=min(children,key=lambda j:np.linalg.norm(self.nodes[j]['center']-p))
            return i
        if self.nodes[leaf]['beta']==1:return leaf
        i=self.nodes[leaf]['parent']
        while self.nodes[i]['children']:
            i=max(self.nodes[i]['children'],key=lambda j:self.nodes[j]['beta'])
        return i

    def forming(self,p,neighbors,r_sense,r_avoid,kappa1=20.,occupancy=None,neighbor_mask=None):
        sensed=self.black_tree.query_ball_point(p,r_sense+self.max_half_diagonal)
        sensed=np.array(sensed,int)
        if len(sensed):
            # Black cells whose boxes intersect the sensing sphere.
            gap=np.maximum(np.abs(self.centers[sensed]-p)-self.sizes[sensed,None]/2,0)
            sensed=sensed[np.linalg.norm(gap,axis=1)<=r_sense]
        inside_map=self.locate(p) is not None
        eta=np.min(np.max(2*np.abs(self.centers[sensed]-p)/self.sizes[sensed,None],axis=1)) if len(sensed) else np.inf
        if inside_map and eta<=1+1e-12:
            vi=np.asarray(self.virtual_tree.query_ball_point(p,r_sense),int)
            if occupancy is not None and len(vi):
                occupied=np.any((occupancy[vi]&neighbor_mask[None])!=0,axis=1)
                vi=vi[~occupied]
            virtual=self.virtual[vi]
            if occupancy is None and len(neighbors) and len(virtual):
                nearest=cKDTree(neighbors).query(virtual)[0]
                virtual=virtual[nearest>=r_avoid/2]
            if not len(virtual):return np.zeros(3)
            dist=np.linalg.norm(virtual-p,axis=1)
            w=.5*(1+np.cos(np.pi*np.minimum(dist/r_sense,1)))
            return kappa1*((virtual-p)*w[:,None]).sum(0)/max(w.sum(),1e-12)
        leaf=self.entering_leaf(p)
        ids=np.array([leaf]) if not inside_map else self.black[sensed]
        if inside_map:
            ids=np.concatenate([np.array([leaf]),ids[ids!=leaf]])
        target=self.all_centers[ids]
        w=self.all_volumes[ids]/np.maximum(np.linalg.norm(target-p,axis=1),1e-12)
        return kappa1*((target-p)*w[:,None]).sum(0)/max(w.sum(),1e-12)

    def velocity(self,p,r_sense,r_avoid,kappa1=20.,kappa2=25.,alpha=0.):
        tree=cKDTree(p);neighbor_ids=tree.query_ball_point(p,r_sense)
        output=np.zeros_like(p)
        blocks=(len(p)+63)//64
        occupancy=np.zeros((len(self.virtual),blocks),np.uint64)
        occupied=self.virtual_tree.query_ball_point(p,r_avoid/2)
        for j,vi in enumerate(occupied):
            vi=np.asarray(vi,int)
            if len(vi):
                vi=vi[np.linalg.norm(self.virtual[vi]-p[j],axis=1)<r_avoid/2]
                occupancy[vi,j//64]|=np.uint64(1)<<np.uint64(j%64)
        for i in range(len(p)):
            ids=np.array([j for j in neighbor_ids[i] if j!=i],int)
            nb=p[ids]
            bitmask=np.zeros(blocks,np.uint64)
            for j in ids:bitmask[j//64]|=np.uint64(1)<<np.uint64(j%64)
            form=self.forming(p[i],nb,r_sense,r_avoid,kappa1,occupancy,bitmask)
            delta=p[i]-nb;dist=np.linalg.norm(delta,axis=1)
            # Isotropic alpha=0 for common point-mass comparison; anisotropic
            # z compensation available only as a separately declared condition.
            radius=r_avoid*np.array([1.,1.,1.+alpha])
            weights=np.maximum(radius[None]/np.maximum(dist[:,None],1e-12)-1,0)
            avoid=kappa2*(weights*delta).sum(0)
            output[i]=form+avoid
        return output
