"""Predeclared simple metric curves for 4/6/8-agent shape screening.

All candidates at the same N have length0.7N metres. This is explicit1D
resource scaling, separate from both old fixed curves and surface-area scaling.
"""
from functools import lru_cache
import numpy as np

KINDS=('line','arch','ring','tilted_ring','ellipse','s_curve','helix','conical_helix','saddle')
CLOSED=frozenset(('ring','tilted_ring','ellipse','saddle'))
LABELS=dict(line='斜直线',arch='竖直拱形',ring='水平圆环',tilted_ring='倾斜圆环',ellipse='倾斜椭圆',
            s_curve='空间S线',helix='单圈螺旋线',conical_helix='锥形螺旋线',saddle='马鞍环')


def parse(shape):
    if not shape.startswith('curve_demo_'):raise ValueError(shape)
    kind,n=shape[len('curve_demo_'):].rsplit('_n',1);n=int(n)
    if kind not in KINDS or n not in (4,6,8):raise ValueError(shape)
    return kind,n


def prototype(kind,t):
    t=np.asarray(t);angle=2*np.pi*t
    if kind=='line':return np.column_stack([2*t-1,.3*(2*t-1),.4*(2*t-1)])
    if kind=='arch':return np.column_stack([-np.cos(np.pi*t),np.zeros_like(t),np.sin(np.pi*t)])
    if kind in ('ring','tilted_ring','ellipse'):
        x=(1.4 if kind=='ellipse' else 1.)*np.cos(angle)
        y=(.8 if kind=='ellipse' else 1.)*np.sin(angle)
        tilt=np.deg2rad(35 if kind!='ring' else 0)
        return np.column_stack([x,y*np.cos(tilt),y*np.sin(tilt)])
    if kind=='s_curve':return np.column_stack([2*t-1,.55*np.sin(angle),.25*np.sin(np.pi*t)])
    if kind in ('helix','conical_helix'):
        radius=.65 if kind=='helix' else .5+.35*t
        return np.column_stack([radius*np.cos(angle),radius*np.sin(angle),1.2*t])
    if kind=='saddle':return np.column_stack([np.cos(angle),np.sin(angle),.3*np.sin(2*angle)])
    raise ValueError(kind)


@lru_cache(maxsize=27)
def transform(kind,n):
    p=prototype(kind,np.linspace(0,1,65537))
    length=np.linalg.norm(np.diff(p,axis=0),axis=1).sum();scale=.7*n/length
    center=(p.min(0)+p.max(0))/2
    height=np.ptp(p[:,2])*scale
    offset=np.array([0.,0.,max(1.2,.6+height/2)])
    return scale,center,offset


def catalog_parametric(shape,t):
    kind,n=parse(shape);scale,center,offset=transform(kind,n)
    return (prototype(kind,t)-center)*scale+offset
