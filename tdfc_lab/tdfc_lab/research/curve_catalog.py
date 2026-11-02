"""Predeclared simple metric curves for 4/6/8-agent shape screening.

All candidates at the same N have length0.7N metres. This is explicit1D
resource scaling, separate from both old fixed curves and surface-area scaling.
"""
from functools import lru_cache
import numpy as np

KINDS=('line','arch','ring','tilted_ring','ellipse','s_curve','helix','conical_helix','saddle','helix_flat',
       'digit2','digit0','digit6','letterL','letterR')
CLOSED=frozenset(('ring','tilted_ring','ellipse','saddle','digit0'))
LABELS=dict(line='斜直线',arch='竖直拱形',ring='水平圆环',tilted_ring='倾斜圆环',ellipse='倾斜椭圆',
            s_curve='空间S线',helix='单圈螺旋线',conical_helix='锥形螺旋线',saddle='马鞍环',
            helix_flat='扁平螺旋线（低竖直跨度）',
            digit2='数字 2',digit0='数字 0',digit6='数字 6',letterL='字母 L',letterR='字母 R')


def parse(shape):
    if not shape.startswith('curve_demo_'):raise ValueError(shape)
    kind,n=shape[len('curve_demo_'):].rsplit('_n',1);n=int(n)
    if kind not in KINDS or n<2:raise ValueError(shape)
    return kind,n


_DIGITS = {
    # x to the right, y up, roughly inside [-1,1]^2
    'digit2': [(-.85,-.90),(.85,-.90),(.45,-.05),(-.30,.40),(-.55,.70),
               (-.45,.90),(-.10,1.00),(.35,.95),(.70,.70),(.80,.40)],
    'letterL': [(-.60,.95),(-.60,-.85),(.60,-.85)],
    'letterR': [(-.60,-.85),(-.60,.95),(.05,.82),(.25,.50),(.05,.18),(-.60,.05),(.60,-.85)],
    'digit6': [(.60,1.00),(.40,.88),(.10,.66),(-.15,.40),(-.33,.12),(-.42,-.12),
               (-.50,-.42),(-.42,-.68),(-.20,-.86),(.10,-.90),(.38,-.78),
               (.55,-.55),(.58,-.28),(.46,-.06),(.22,.06),(-.02,.06),(-.24,-.04),
               (-.38,-.16)],
}


def _polyline(points, t):
    p = np.asarray(points, float)
    seg = np.linalg.norm(np.diff(p, axis=0), axis=1)
    s = np.r_[0., np.cumsum(seg)]
    d = np.asarray(t, float) * s[-1]
    return np.column_stack([np.interp(d, s, p[:, 0]), np.interp(d, s, p[:, 1]),
                            np.zeros_like(d)])


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
    if kind in ('helix','conical_helix','helix_flat'):
        if kind=='conical_helix':radius=.5+.35*t
        elif kind=='helix_flat':radius=.85
        else:radius=.65
        rise=.6 if kind=='helix_flat' else 1.2
        return np.column_stack([radius*np.cos(angle),radius*np.sin(angle),rise*t])
    if kind=='saddle':return np.column_stack([np.cos(angle),np.sin(angle),.3*np.sin(2*angle)])
    if kind=='letterL':return _polyline(_DIGITS['letterL'],t)
    if kind=='letterR':return _polyline(_DIGITS['letterR'],t)
    if kind=='digit2':return _polyline(_DIGITS['digit2'],t)
    if kind=='digit6':return _polyline(_DIGITS['digit6'],t)
    if kind=='digit0':return np.column_stack([.62*np.cos(angle),1.0*np.sin(angle),np.zeros_like(angle)])
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
