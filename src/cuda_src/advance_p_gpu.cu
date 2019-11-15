#define IN_spa

#include <cub/cub.cuh>

#include "advance_p_gpu.cuh"
#include "gpu_util.cuh"
#include "move_p_gpu.cuh"

__global__ void handle_particle_movers(handle_args args, int temp_nm) {
  const int block_rank = blockIdx.x;
  const int n_block = gridDim.x;
  const int thread_rank = threadIdx.x;
  const int n_thread = blockDim.x;
  const int stride_size = args.stride_size;

  const float qsp = args.qsp;
  int itmp, n;
  int *nm = args.nm;

  GPU_DISTRIBUTE(temp_nm, stride_size, block_rank, itmp, n);
  if (thread_rank < n) {
    particle_mover_t pm = args.temp_pm_array[itmp + thread_rank];
    particle_t p = args.p0[pm.i];

    if (move_p_gpu(&p, &pm,
                   args.a0, args.g_neighbor,
                   args.g_rangel, args.g_rangeh, qsp))  // Unlikely
    {
      // assume max_nm is large enough
      int the = atomicAdd(nm, 1);
      args.pm_array[the] = pm;
    }

    args.p0[pm.i] = p;
  }
}

#define one 1.f

__global__ void advance_p_gpu(advance_p_gpu_args args) {
  const int bstart = blockIdx.x * args.stride_size;
  const int bend = min((blockIdx.x + 1) * args.stride_size, args.np);

  const float one_third = 1.f / 3.f;
  const float two_fifteenths = 2.f / 15.f;

  for (int pid = bstart + threadIdx.x; pid < bend; pid += blockDim.x) {
      particle_t& p = args.p0[pid];
      const interpolator_t& f = args.f0[p.i];

      register float4 dxyz = reinterpret_cast<float4*>(&p.dx)[0];
      float& dx = dxyz.x;  // Load position
      float& dy = dxyz.y;
      float& dz = dxyz.z;

      float hax = args.qdt_2mc * ((f.ex + dy * f.dexdy) + dz * (f.dexdz + dy * f.d2exdydz));
      float hay = args.qdt_2mc * ((f.ey + dz * f.deydz) + dx * (f.deydx + dz * f.d2eydzdx));
      float haz = args.qdt_2mc * ((f.ez + dx * f.dezdx) + dy * (f.dezdy + dx * f.d2ezdxdy));

      float cbx = f.cbx + dx * f.dcbxdx;  // Interpolate B
      float cby = f.cby + dy * f.dcbydy;
      float cbz = f.cbz + dz * f.dcbzdz;

      register float4 uxyz = reinterpret_cast<float4*>(&p.ux)[0];
      float& ux = uxyz.x;  // Load momentum
      float& uy = uxyz.y;
      float& uz = uxyz.z;
      float& q  = uxyz.w;
      
      ux += hax;  // Half advance E
      uy += hay;
      uz += haz;

      float v0 = args.qdt_2mc * rsqrtf(one + (ux * ux + (uy * uy + uz * uz)));

      // Boris - scalars
      float v1 = cbx * cbx + (cby * cby + cbz * cbz);
      float v2 = (v0 * v0) * v1;
      float v3 = v0 * (one + v2 * (one_third + v2 * two_fifteenths));
      float v4 = v3 / (one + v1 * (v3 * v3));
      v4 += v4;

      v0 = ux + v3 * (uy * cbz - uz * cby);  // Boris - uprime
      v1 = uy + v3 * (uz * cbx - ux * cbz);
      v2 = uz + v3 * (ux * cby - uy * cbx);

      ux += v4 * (v1 * cbz - v2 * cby);  // Boris - rotation
      uy += v4 * (v2 * cbx - v0 * cbz);
      uz += v4 * (v0 * cby - v1 * cbx);

      ux += hax;  // Half advance E
      uy += hay;
      uz += haz;

      p.ux = ux;  // Store new position
      p.uy = uy;
      p.uz = uz;

      v0 = rsqrtf(one + (ux * ux + (uy * uy + uz * uz)));

      // Get norm displacement

      ux *= args.cdt_dx;
      uy *= args.cdt_dy;
      uz *= args.cdt_dz;

      ux *= v0;
      uy *= v0;
      uz *= v0;

      v0 = dx + ux;  // Streak midpoint (inbnds)
      v1 = dy + uy;
      v2 = dz + uz;

      v3 = v0 + ux;  // New position
      v4 = v1 + uy;
      float v5 = v2 + uz;

      // FIXME-KJB: COULD SHORT CIRCUIT ACCUMULATION IN THE CASE WHERE QSP==0!
      if (v3 <= one && v4 <= one && v5 <= one &&  // Check if inbnds
          -v3 <= one && -v4 <= one && -v5 <= one) {
        // Common case (inbnds).  Note: accumulator values are 4 times
        // the total physical charge that passed through the appropriate
        // current quadrant in a time-step.

        // float q = p.w;
        q *= args.qsp;

        p.dx = v3;  // Store new position
        p.dy = v4;
        p.dz = v5;
        // reinterpret_cast<float2*>(&p.dx)[0] = make_float2(v3, v4);
        // p.dz = v5;

        dx = v0;  // Streak midpoint
        dy = v1;
        dz = v2;

        v5 = q * ux * uy * uz * one_third;  // Compute correction

        float *a = (float *)(args.a0 + p.i);  // Get accumulator

#define ACCUMULATE_J(X, Y, Z, offset)                         \
  v4 = q * u##X;   /* v2 = q ux                            */ \
  v1 = v4 * d##Y;  /* v1 = q ux dy                         */ \
  v0 = v4 - v1;    /* v0 = q ux (1-dy)                     */ \
  v1 += v4;        /* v1 = q ux (1+dy)                     */ \
  v4 = one + d##Z; /* v4 = 1+dz                            */ \
  v2 = v0 * v4;    /* v2 = q ux (1-dy)(1+dz)               */ \
  v3 = v1 * v4;    /* v3 = q ux (1+dy)(1+dz)               */ \
  v4 = one - d##Z; /* v4 = 1-dz                            */ \
  v0 *= v4;        /* v0 = q ux (1-dy)(1-dz)               */ \
  v1 *= v4;        /* v1 = q ux (1+dy)(1-dz)               */ \
  v0 += v5;        /* v0 = q ux [ (1-dy)(1-dz) + uy*uz/3 ] */ \
  v1 -= v5;        /* v1 = q ux [ (1+dy)(1-dz) - uy*uz/3 ] */ \
  v2 -= v5;        /* v2 = q ux [ (1-dy)(1+dz) - uy*uz/3 ] */ \
  v3 += v5;        /* v3 = q ux [ (1+dy)(1+dz) + uy*uz/3 ] */ \
  atomicAdd(a + offset + 0, v0);                              \
  atomicAdd(a + offset + 1, v1);                              \
  atomicAdd(a + offset + 2, v2);                              \
  atomicAdd(a + offset + 3, v3);

        ACCUMULATE_J(x, y, z, 0);
        ACCUMULATE_J(y, z, x, 4);
        ACCUMULATE_J(z, x, y, 8);

#undef ACCUMULATE_J

      } else {
        particle_mover_t pm;
        pm.dispx = ux;
        pm.dispy = uy;
        pm.dispz = uz;
        pm.i = pid;
        int the = atomicAdd(args.nm, 1);
        reinterpret_cast<float4*>(args.temp_pm_array)[the] = reinterpret_cast<float4*>(&pm)[0];
      }
  }
}
