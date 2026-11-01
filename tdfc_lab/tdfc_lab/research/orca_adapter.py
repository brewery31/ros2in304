"""ctypes adapter: upstream 3D ORCA, not a reimplementation."""
import ctypes
from pathlib import Path
import numpy as np
_LIB=None

def orca_velocity(p,v,preferred,*,dt,r_vis,d_safe,v_max,horizon=.4,margin=.02,nonfinite_policy='raise'):
    global _LIB
    if _LIB is None:
        path=Path(__file__).resolve().parent/'liborca.so'
        if not path.exists():
            from ament_index_python.packages import get_package_share_directory
            path=Path(get_package_share_directory('tdfc_lab'))/'lib/liborca.so'
        _LIB=ctypes.CDLL(str(path))
        ptr=np.ctypeslib.ndpointer(dtype=np.float64,flags='C_CONTIGUOUS')
        _LIB.orca_step.argtypes=[ctypes.c_int,ptr,ptr,ptr]+[ctypes.c_double]*5+[ptr,ptr]
        _LIB.orca_step.restype=ctypes.c_int
    p,v,preferred=[np.ascontiguousarray(x,dtype=np.float64) for x in (p,v,preferred)]
    if p.shape != v.shape or p.shape != preferred.shape or p.ndim !=2 or p.shape[1]!=3:
        raise ValueError('expected matching N x 3 arrays')
    if nonfinite_policy not in ('raise','ego_brake'):raise ValueError('unknown ORCA nonfinite policy')
    if not all(np.isfinite(x).all() for x in (p,v,preferred)):
        raise ValueError('ORCA inputs must be finite')
    out=np.empty_like(p); residual=np.zeros(len(p))
    status=_LIB.orca_step(len(p),p,v,preferred,dt,r_vis,horizon,(d_safe+margin)/2,v_max,out,residual)
    if status:
        raise RuntimeError('upstream ORCA failed')
    bad=(~np.isfinite(out).all(axis=1))|(~np.isfinite(residual))
    if np.any(bad):
        if nonfinite_policy=='raise':raise RuntimeError('upstream ORCA failed')
        # Each affected ego requests executable zero-setpoint braking. The
        # caller MUST certify this with its physical stopping governor. Keep
        # NaN residuals as explicit per-ego failure markers for logging.
        out[bad]=0.;residual[bad]=np.nan
    return out,residual
