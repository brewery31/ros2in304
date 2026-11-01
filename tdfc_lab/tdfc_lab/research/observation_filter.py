"""Causal constant-velocity measurement smoothing, for empirical research only.

Inputs are observations and their timestamps. No simulator truth, target shape,
neighbor commands or future measurements enter this object. A propagated estimate
is not an exact or bounded-error observation and does not certify collision safety.
"""
import numpy as np


class CausalCVFilter:
    def __init__(self, *, smoothing_seconds, predict_to_now=True, v_max):
        if not np.isfinite(smoothing_seconds) or smoothing_seconds < 0:
            raise ValueError('smoothing_seconds must be finite and nonnegative')
        if not isinstance(predict_to_now, bool):
            raise ValueError('predict_to_now must be boolean')
        if not np.isfinite(v_max) or v_max <= 0:
            raise ValueError('v_max must be finite and positive')
        self.tau=float(smoothing_seconds);self.predict=predict_to_now;self.v_max=float(v_max)
        self.position=None;self.velocity=None;self.time=None;self.ignored_frames=0

    def update(self, position, velocity, *, measurement_time, decision_time):
        pm=np.asarray(position,dtype=float);vm=np.asarray(velocity,dtype=float)
        if pm.ndim!=2 or pm.shape[1]!=3 or vm.shape!=pm.shape:
            raise ValueError('position and velocity must be matching N by3 arrays')
        if not np.isfinite(pm).all() or not np.isfinite(vm).all():
            raise ValueError('nonfinite observation')
        if not np.isfinite(measurement_time) or not np.isfinite(decision_time) or measurement_time>decision_time+1e-12:
            raise ValueError('invalid observation timestamps')
        if self.position is not None and pm.shape!=self.position.shape:
            raise ValueError('tracked ID ordering and count must stay fixed')
        if self.time is not None and measurement_time<self.time-1e-12:
            raise ValueError('out-of-order measurement')
        bounded=vm*np.minimum(1.,self.v_max/np.maximum(np.linalg.norm(vm,axis=1),1e-30))[:,None]
        if self.position is None:
            self.position=pm.copy();self.velocity=bounded.copy();self.time=float(measurement_time)
        elif measurement_time<=self.time+1e-12:
            # A repeated old frame must not be counted as a new independent sample.
            self.ignored_frames+=1
        else:
            elapsed=measurement_time-self.time
            alpha=1. if self.tau==0 else -np.expm1(-elapsed/self.tau)
            predicted=self.position+elapsed*self.velocity
            self.position=predicted+alpha*(pm-predicted)
            self.velocity=self.velocity+alpha*(bounded-self.velocity)
            self.time=float(measurement_time)
        age=decision_time-self.time if self.predict else 0.
        return self.position.copy()+age*self.velocity,self.velocity.copy()
