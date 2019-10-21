#ifndef __NTHU_MOVE_P_GPU_H__
#define __NTHU_MOVE_P_GPU_H__

#include "gpu.cuh"

__device__ 
int move_p_gpu(particle_t *p_register, particle_mover_t *pm,
    accumulator_t *a0, const int64_t *g_neighbor,
    int64_t g_rangel, int64_t g_rangeh, const float qsp) {

        float s_midx, s_midy, s_midz;
        float s_dispx, s_dispy, s_dispz;
        float s_dir[3];
        float v0, v1, v2, v3, v4, v5, q;
        int axis, face;
        int64_t neighbor;
        float * a;
        particle_t *p = p_register;

        q = qsp * p->w;
      
        for(;;) {
          s_midx = p->dx;
          s_midy = p->dy;
          s_midz = p->dz;
      
          s_dispx = pm->dispx;
          s_dispy = pm->dispy;
          s_dispz = pm->dispz;
      
          s_dir[0] = (s_dispx>0.0f) ? 1.0f : -1.0f;
          s_dir[1] = (s_dispy>0.0f) ? 1.0f : -1.0f;
          s_dir[2] = (s_dispz>0.0f) ? 1.0f : -1.0f;
      
          // Compute the twice the fractional distance to each potential
          // streak/cell face intersection.
          v0 = (s_dispx==0.0f) ? 3.4e38f : (s_dir[0]-s_midx)/s_dispx;
          v1 = (s_dispy==0.0f) ? 3.4e38f : (s_dir[1]-s_midy)/s_dispy;
          v2 = (s_dispz==0.0f) ? 3.4e38f : (s_dir[2]-s_midz)/s_dispz;
      

          /**/      v3=2.0f, axis=3;
          if(v0<v3) v3=v0,   axis=0;
          if(v1<v3) v3=v1,   axis=1;
          if(v2<v3) v3=v2,   axis=2;
          v3 *= 0.5f;
      
          // Compute the midpoint and the normalized displacement of the streak
          s_dispx *= v3;
          s_dispy *= v3;
          s_dispz *= v3;
          s_midx += s_dispx;
          s_midy += s_dispy;
          s_midz += s_dispz;
      
          // Accumulate the streak.  Note: accumulator values are 4 times
          // the total physical charge that passed through the appropriate
          // current quadrant in a time-step
          v5 = q*s_dispx*s_dispy*s_dispz*(1./3.);
          a = (float *)(a0 + p->i);
          
      #   define accumulate_j(X,Y,Z)                                        \
          v4  = q*s_disp##X;    /* v2 = q ux                            */  \
          v1  = v4*s_mid##Y;    /* v1 = q ux dy                         */  \
          v0  = v4-v1;          /* v0 = q ux (1-dy)                     */  \
          v1 += v4;             /* v1 = q ux (1+dy)                     */  \
          v4  = 1+s_mid##Z;     /* v4 = 1+dz                            */  \
          v2  = v0*v4;          /* v2 = q ux (1-dy)(1+dz)               */  \
          v3  = v1*v4;          /* v3 = q ux (1+dy)(1+dz)               */  \
          v4  = 1-s_mid##Z;     /* v4 = 1-dz                            */  \
          v0 *= v4;             /* v0 = q ux (1-dy)(1-dz)               */  \
          v1 *= v4;             /* v1 = q ux (1+dy)(1-dz)               */  \
          v0 += v5;             /* v0 = q ux [ (1-dy)(1-dz) + uy*uz/3 ] */  \
          v1 -= v5;             /* v1 = q ux [ (1+dy)(1-dz) - uy*uz/3 ] */  \
          v2 -= v5;             /* v2 = q ux [ (1-dy)(1+dz) - uy*uz/3 ] */  \
          v3 += v5;             /* v3 = q ux [ (1+dy)(1+dz) + uy*uz/3 ] */  \
          atomicAdd(a + 0, v0);                                                       \
          atomicAdd(a + 1, v1);                                                       \
          atomicAdd(a + 2, v2);                                                       \
          atomicAdd(a + 3, v3)

          accumulate_j(x,y,z); a += 4;
          accumulate_j(y,z,x); a += 4;
          accumulate_j(z,x,y);

      #   undef accumulate_j
      
          // Compute the remaining particle displacment
          pm->dispx -= s_dispx;
          pm->dispy -= s_dispy;
          pm->dispz -= s_dispz;
      
          // Compute the new particle offset
          p->dx += s_dispx+s_dispx;
          p->dy += s_dispy+s_dispy;
          p->dz += s_dispz+s_dispz;
      
          // If an end streak, return success (should be ~50% of the time)
      
          if( axis==3 ) break;
      
          // Determine if the particle crossed into a local cell or if it
          // hit a boundary and convert the coordinate system accordingly.
          // Note: Crossing into a local cell should happen ~50% of the
          // time; hitting a boundary is usually a rare event.  Note: the
          // entry / exit coordinate for the particle is guaranteed to be
          // +/-1 _exactly_ for the particle.
      
          v0 = s_dir[axis];
          (&(p->dx))[axis] = v0; 
          face = axis; if( v0>0 ) face += 3;
          neighbor = g_neighbor[ 6*p->i + face ];
      
          if(  neighbor==reflect_particles  ) { // unlikely
            (&(p->ux    ))[axis] = -(&(p->ux    ))[axis];
            (&(pm->dispx))[axis] = -(&(pm->dispx))[axis];
            continue;
          }
      
          if( neighbor<g_rangel || neighbor>g_rangeh  ) { // unlikely
            p->i = 8*p->i + face;
            return 1; // Return "mover still in use"
          }

          p->i = neighbor - g_rangel; // Compute local index of neighbor
          (&(p->dx))[axis] = -v0;      // Convert coordinate system
        }
        
        return 0;
}


#endif