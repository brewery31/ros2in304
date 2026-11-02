"""Public letter stroke priors. No robot identities, assignment or target slots.

Branches are separate edges: never connect pen lifts by artificial diagonals.
Scale uses the same total arclength rule as the curve experiments: L=0.7N.
"""
import numpy as np
from .geometry import PointCloud


def parse_letter(shape):
    if not shape.startswith('curve_letter_'):
        raise ValueError('expected curve_letter_R/A/L_n5')
    letter, count = shape[len('curve_letter_'):].rsplit('_n',1)
    n = int(count)
    if letter not in ('R','A','L') or n != 5:
        raise ValueError('letter experiments require R/A/L and five agents')
    return letter,n


def stroke_graph(letter, agents=5):
    if letter not in ('R','A','L') or agents != 5:
        raise ValueError('R/A/L five-agent prior required')
    definitions = {
        'R': ([(-.7,-.7),(-.7,.7),(.7,.7),(.7,0.),(-.7,0.),(.7,-.7)],
              [(0,4),(4,1),(1,2),(2,3),(3,4),(4,5)]),
        'A': ([(-.7,-.7),(0.,.7),(.7,-.7),(-.35,0.),(.35,0.)],
              [(0,3),(3,1),(1,4),(4,2),(3,4)]),
        'L': ([(-.7,.7),(-.7,-.7),(.7,-.7)],[(0,1),(1,2)]),
    }
    points, edges = definitions[letter]
    xy = np.array(points,float)
    vertices = np.column_stack([xy[:,0],np.zeros(len(xy)),xy[:,1]])
    edges = np.array(edges,int)
    lengths = np.linalg.norm(vertices[edges[:,1]]-vertices[edges[:,0]],axis=1)
    vertices *= .7*agents/lengths.sum()
    vertices -= (vertices.min(0)+vertices.max(0))/2
    vertices[:,2] += max(1.2,.6+np.ptp(vertices[:,2])/2)
    return vertices,edges


def stroke_geometry(shape, control_points=4096, eval_points=32768):
    letter,n = parse_letter(shape)
    vertices,edges = stroke_graph(letter,n)
    delta = vertices[edges[:,1]]-vertices[edges[:,0]]
    lengths = np.linalg.norm(delta,axis=1)
    tangents = delta/lengths[:,None]
    degree = np.bincount(edges.ravel(),minlength=len(vertices))

    def sample(count, evaluation):
        if count < 2*len(edges):raise ValueError('at least two samples per stroke required')
        allocation=(count-2*len(edges))*lengths/lengths.sum()
        counts=2+np.floor(allocation).astype(int)
        order=np.argsort(-(allocation-np.floor(allocation)),kind='stable')
        counts[order[:count-counts.sum()]]+=1
        pp=[];tt=[];weights=[];endpoint=[];inward=[]
        for k,((a,b),m) in enumerate(zip(edges,counts)):
            t=(np.arange(m)+.5)/m if evaluation else np.linspace(0,1,m)
            pp.append(vertices[a]+t[:,None]*delta[k])
            tt.append(np.tile(tangents[k],(m,1)))
            weights.extend(np.full(m,lengths[k]/m))
            leaf=((t==0)&(degree[a]==1))|((t==1)&(degree[b]==1))
            endpoint.extend(leaf)
            inward.append(np.where((t==1)[:,None],-tangents[k],tangents[k]))
        cloud=PointCloud.from_points(np.concatenate(pp),area=0.,name=shape)
        cloud.meta.update(dimension=1,length=float(lengths.sum()),tangents=np.concatenate(tt),
                          quadrature_weights=np.array(weights),endpoint_mask=np.array(endpoint),
                          endpoint_inward=np.concatenate(inward),closed_curve=False,
                          stroke_vertices=vertices.copy(),stroke_edges=edges.copy(),
                          branched_stroke_prior=True,no_target_assignment=True,
                          length_rule='0.7N metres, same rule as ring/helix')
        return cloud
    return sample(control_points,False),sample(eval_points,True)
