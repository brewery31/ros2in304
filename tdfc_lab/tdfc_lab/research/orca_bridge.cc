// Project adapter to unmodified Apache-2.0 RVO2-3D. See third_party/RVO2-3D.
#include "RVO.h"
#include <algorithm>
#include <cstddef>
extern "C" int orca_step(int n, const double* p, const double* v,
 const double* pref, double dt, double range, double horizon, double radius,
 double vmax, double* result, double* residual) {
 try {
  RVO::RVOSimulator sim(dt,range,n,horizon,radius,vmax);
  for(int i=0;i<n;++i){
   sim.addAgent(RVO::Vector3(p[3*i],p[3*i+1],p[3*i+2]));
   sim.setAgentVelocity(i,RVO::Vector3(v[3*i],v[3*i+1],v[3*i+2]));
   sim.setAgentPrefVelocity(i,RVO::Vector3(pref[3*i],pref[3*i+1],pref[3*i+2]));
  }
  sim.doStep();
  for(int i=0;i<n;++i){
   const RVO::Vector3 w=sim.getAgentVelocity(i);
   for(int k=0;k<3;++k)result[3*i+k]=w[k];
   residual[i]=0;
   for(std::size_t j=0;j<sim.getAgentNumORCAPlanes(i);++j){
    const RVO::Plane plane=sim.getAgentORCAPlane(i,j);
    residual[i]=std::max(residual[i],double(plane.normal*(plane.point-w)));
   }
  }
  return 0;
 } catch(...) { return -1; }
}
