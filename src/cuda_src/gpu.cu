#include <CppToolkit/color.h>

#include <algorithm>
#include <cub/cub.cuh>
#include <cub/util_allocator.cuh>
#include <map>

#include "accumulate_rho_p_gpu.cuh"
#include "advance_p_gpu.cuh"
#include "backfill_gpu.cuh"
#include "energy_p_gpu.cuh"
#include "gpu.cuh"
#include "gpu_util.cuh"
#include "sort_p_gpu.cuh"

namespace vpic_gpu {

enum { MAX_PBC = 32,
       MAX_SP = 32 };

gpu_memory_allocator gm;
cudaStream_t sp_streams[MAX_SP];
const int DEFAULT_NUM_THREAD = 32;
const int DEFAULT_NP_STRIDE_SIZE = 4096;
const int DEFAULT_NM_STRIDE_SIZE = 256;

void mpiSetDevice(int rank) {
  cudaSetDevice(rank);
}

void cudaInitSpeciesStream(species_t *sp_list) {
  species_t *sp;
  LIST_FOR_EACH(sp, sp_list) {
    MY_MESSAGE(("Creating cuda stream on %s at id:%d", sp->name, sp->id));
    cudaStreamCreate(&sp_streams[sp->id]);
  }
}

void sort_p_gpu_launcher(species_t *sp) {
  const int num_threads = 32;
  const int num_blocks = 512;

  if(sp->np <= 0) return;
  particle_t *device_p = (particle_t *)gm.map_to_device(sp->p, sizeof(particle_t) * sp->max_np);

  int num_items = sp->np;

  cub::DoubleBuffer<int32_t> d_keys;
  cub::DoubleBuffer<particle_t> d_values;
  cub::CachingDeviceAllocator g_allocator(true);

  gpuErrchk(g_allocator.DeviceAllocate((void **)&d_keys.d_buffers[0], sizeof(int32_t) * num_items));
  gpuErrchk(g_allocator.DeviceAllocate((void **)&d_keys.d_buffers[1], sizeof(int32_t) * num_items));
  d_values.d_buffers[0] = device_p;
  gpuErrchk(g_allocator.DeviceAllocate((void **)&d_values.d_buffers[1], sizeof(particle_t) * num_items));

  // Allocate temporary storage
  size_t temp_storage_bytes = 0;
  void *d_temp_storage = NULL;

  gpuErrchk(cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, d_keys, d_values, num_items));
  gpuErrchk(g_allocator.DeviceAllocate(&d_temp_storage, temp_storage_bytes));

  // Initialize device arrays
  copy_particle_index<<<num_blocks, num_threads>>>(d_keys.d_buffers[0], device_p, num_items);

  // Run
  gpuErrchk(cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, d_keys, d_values, num_items));

  // Copy particles back
  gpuErrchk(cudaMemcpy(device_p, d_values.d_buffers[1], sizeof(particle_t) * num_items, cudaMemcpyDeviceToDevice));

  // Cleanup
  if (d_keys.d_buffers[0]) gpuErrchk(g_allocator.DeviceFree(d_keys.d_buffers[0]));
  if (d_keys.d_buffers[1]) gpuErrchk(g_allocator.DeviceFree(d_keys.d_buffers[1]));
  if (d_values.d_buffers[1]) gpuErrchk(g_allocator.DeviceFree(d_values.d_buffers[1]));
  if (d_temp_storage) gpuErrchk(g_allocator.DeviceFree(d_temp_storage));
}

void advance_p_gpu_launcher(species_t *sp_list, accumulator_array_t *aa, const interpolator_array_t *ia) {
  int num_threads, num_blocks, stride_size;
  advance_p_gpu_args gpu_args;
  handle_args hgpu_args;
  species_t *sp;

  gpu_args.a0 = (accumulator_t *)gm.map_to_device(aa->a, sizeof(accumulator_t) * aa->g->nv);
  hgpu_args.a0 = gpu_args.a0;
  cudaMemset(gpu_args.a0, 0, sizeof(accumulator_t) * aa->g->nv);  // clear accumulator array
  gpu_args.f0 = (interpolator_t *)gm.copy_to_device((host_pointer)ia->i, sizeof(interpolator_t) * ia->g->nv);
  hgpu_args.f0 = gpu_args.f0;

  // std::vector<advance_p_gpu_args> args(MAX_SP);
  std::vector<handle_args> args(MAX_SP);
  LIST_FOR_EACH(sp, sp_list) {
    if(sp->np <= 0) continue;
    gpu_args.qdt_2mc = (sp->q * sp->g->dt) / (2 * sp->m * sp->g->cvac);
    gpu_args.cdt_dx = sp->g->cvac * sp->g->dt * sp->g->rdx;
    gpu_args.cdt_dy = sp->g->cvac * sp->g->dt * sp->g->rdy;
    gpu_args.cdt_dz = sp->g->cvac * sp->g->dt * sp->g->rdz;
    gpu_args.qsp = sp->q;
    gpu_args.np = sp->np;

    hgpu_args.qdt_2mc = (sp->q * sp->g->dt) / (2 * sp->m * sp->g->cvac);
    hgpu_args.cdt_dx = sp->g->cvac * sp->g->dt * sp->g->rdx;
    hgpu_args.cdt_dy = sp->g->cvac * sp->g->dt * sp->g->rdy;
    hgpu_args.cdt_dz = sp->g->cvac * sp->g->dt * sp->g->rdz;
    hgpu_args.qsp = sp->q;
    hgpu_args.np = sp->np;
    hgpu_args.g_rangel = sp->g->rangel;
    hgpu_args.g_rangeh = sp->g->rangeh;
    hgpu_args.g_neighbor = (int64_t *)gm.map_to_device(sp->g->neighbor, sizeof(int64_t) * sp->g->nv * 6);

    gpu_args.pm_array = (particle_mover_t *)gm.map_to_device(sp->pm, sizeof(particle_mover_t) * sp->max_nm);
    gpu_args.temp_pm_array = (particle_mover_t *)gm.map_to_device(sp->pm, sizeof(particle_mover_t) * sp->max_nm);
    gpu_args.p0 = (particle_t *)gm.map_to_device(sp->p, sizeof(particle_t) * sp->max_np);
    hgpu_args.pm_array = gpu_args.pm_array;
    hgpu_args.temp_pm_array = gpu_args.temp_pm_array;
    hgpu_args.p0 = gpu_args.p0;

    sp->nm = 0;
    gpu_args.nm = (int *)gm.copy_to_device(&sp->nm, sizeof(int));
    hgpu_args.nm = gpu_args.nm;

    // stride_size = DEFAULT_NP_STRIDE_SIZE;
    stride_size = 2048;
    // num_threads = DEFAULT_NUM_THREAD;
    num_threads = 128;
    num_blocks = MATH_CEIL(sp->np, stride_size);
    gpu_args.stride_size = stride_size;
    hgpu_args.stride_size = stride_size;

    // args[sp->id] = gpu_args;
    args[sp->id] = hgpu_args;
    // advance_p_gpu<<<num_blocks, num_threads, 0, sp_streams[sp->id]>>>(args[sp->id]);
    advance_p_gpu<<<num_blocks, num_threads, 0, sp_streams[sp->id]>>>(gpu_args);
  }

  LIST_FOR_EACH(sp ,sp_list){
    if(sp->np <= 0) continue;
    // gm.copy_to_host(&sp->nm, sizeof(int));
    // cudaStreamSynchronize(sp_streams[sp->id]);
    int *device_nm = (int *)gm.map_to_device(&sp->nm, sizeof(int));
    cudaMemcpyAsync(&sp->nm, device_nm, sizeof(int), cudaMemcpyDeviceToHost, sp_streams[sp->id]);
    /*************************************************************/
    // MY_MESSAGE(("%s->nm: %d", sp->name, sp->nm));
    int temp_nm = sp->nm;
    sp->nm = 0;
    cudaMemsetAsync(device_nm, 0, sizeof(int), sp_streams[sp->id]);
    // gpu_args.nm = (int *)gm.copy_to_device(&sp->nm, sizeof(int));

    stride_size = DEFAULT_NM_STRIDE_SIZE;
    if(temp_nm > 0){
        num_threads = min(temp_nm, stride_size);
        num_blocks = MATH_CEIL(temp_nm, stride_size);
        args[sp->id].stride_size = stride_size;
        handle_particle_movers<<<num_blocks, num_threads, 0, sp_streams[sp->id]>>>(args[sp->id], temp_nm);
    }
    // handle_particle_movers<<<num_blocks, num_threads>>>(gpu_args, temp_nm);
    // cudaMemsetAsync(device_nm, 0, sizeof(int), sp_streams[sp->id]);
    cudaMemcpyAsync(&sp->nm, device_nm, sizeof(int), cudaMemcpyDeviceToHost, sp_streams[sp->id]);
  }

  LIST_FOR_EACH(sp, sp_list){
    cudaStreamSynchronize(sp_streams[sp->id]);
    // gm.copy_to_host(&sp->nm, sizeof(int));
  }
  gm.copy_to_host(aa->a, sizeof(accumulator_t) * aa->g->nv);
}

void boundary_p_get_p_pm(particle_t *p0, particle_mover_t *pm, species_t *sp) {
  int num_threads, num_blocks, stride_size;

  int np = sp->np;
  int nm = sp->nm;

  if (nm <= 0 || np <= 0) {
    return;
  }

  particle_t *device_p = (particle_t *)gm.map_to_device(sp->p, sizeof(particle_t) * sp->max_np);
  particle_mover_t *device_pm = (particle_mover_t *)gm.map_to_device(sp->pm, sizeof(particle_mover_t) * sp->nm);

  //allocate device p0/pm
  particle_t *device_preload_p;
  particle_t *device_particle_temp;
  int *device_particle_counter;
  gpuErrchk(cudaMalloc(&device_preload_p, nm * sizeof(particle_t)));
  gpuErrchk(cudaMalloc(&device_particle_temp, nm * sizeof(particle_t)));

  gpuErrchk(cudaMalloc(&device_particle_counter, sizeof(int)));
  gpuErrchk(cudaMemset(device_particle_counter, 0, sizeof(int)));

  //find all device p0/pm with back-filling
  stride_size = DEFAULT_NM_STRIDE_SIZE;
  num_threads = min(nm, stride_size);
  num_blocks = MATH_CEIL(nm, stride_size);
  back_fill_stage_1<<<num_blocks, num_threads>>>(device_p, device_pm,
                                                 device_particle_temp, device_particle_counter,
                                                 device_preload_p, np, nm, stride_size);
  back_fill_stage_2<<<num_blocks, num_threads>>>(device_p, device_pm,
                                                 device_particle_temp, device_particle_counter,
                                                 device_preload_p, np, nm, stride_size);
  back_fill_stage_3<<<num_blocks, num_threads>>>(device_p, device_pm,
                                                 device_particle_temp, device_particle_counter,
                                                 device_preload_p, np, nm, stride_size);

  //transfer device to host
  gpuErrchk(cudaMemcpy(p0, device_preload_p, nm * sizeof(particle_t), cudaMemcpyDeviceToHost));
  gpuErrchk(cudaMemcpy(pm, device_pm, nm * sizeof(particle_mover_t), cudaMemcpyDeviceToHost));

  //free device p0/pm
  gpuErrchk(cudaFree(device_preload_p));
  gpuErrchk(cudaFree(device_particle_temp));
  gpuErrchk(cudaFree(device_particle_counter));
}

__global__ void check_p_idx(int Device, particle_t *device_p, int nv, int np) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < np) {
    particle p = device_p[idx];
    if (p.i < 0) {
      printf(RED "[%d]check_p_idx: ERROR idx %d < 0 at %d; dx = %f" COLOR_END "\n", Device, p.i, idx, p.dx);
    }
    if (p.i >= nv) {
      printf(RED "[%d]check_p_idx: ERROR idx %d > nv at %d; dx = %f" COLOR_END "\n", Device, p.i, idx, p.dx);
    }
  }
}

void append_p_and_pm(particle_t *temp_p, particle_mover_t *temp_pm,
                     int pi_cnt, int pm_cnt, species_t *sp) {
  int np = sp->np;
  int nm = sp->nm;

  if (nm <= 0 || np <= 0) {
    return;
  }

  particle_t *device_p = (particle_t *)gm.map_to_device(sp->p, sizeof(particle_t) * sp->max_np);
  particle_mover_t *device_pm = (particle_mover_t *)gm.map_to_device(sp->pm, sizeof(particle_mover_t) * sp->nm);

  if (pi_cnt) gpuErrchk(cudaMemcpy(device_p + sp->np - pi_cnt, temp_p, pi_cnt * sizeof(particle_t), cudaMemcpyHostToDevice));
  if (pm_cnt) gpuErrchk(cudaMemcpy(device_pm + sp->nm - pm_cnt, temp_pm, pm_cnt * sizeof(particle_mover_t), cudaMemcpyHostToDevice));
}

void accumulate_rho_p_gpu_launcher(field_array_t *fa, const species_t *sp) {
  if(sp->np <= 0) return;
  int num_threads, num_blocks, stride_size;
  accumulate_rho_p_gpu_args gpu_args;

  gpu_args.q_8V = sp->q * sp->g->r8V;
  gpu_args.np = sp->np;
  gpu_args.sy = sp->g->sy;
  gpu_args.sz = sp->g->sz;

  gpu_args.p = (particle_t *)gm.map_to_device(sp->p, sizeof(particle_t) * sp->max_np);
  gpu_args.f = (field_t *)gm.copy_to_device(fa->f, sizeof(field_t) * sp->g->nv);

  stride_size = DEFAULT_NP_STRIDE_SIZE;
  num_threads = DEFAULT_NUM_THREAD;
  num_blocks = MATH_CEIL(sp->np, stride_size);
  gpu_args.stride_size = stride_size;
  accumulate_rho_p_gpu<<<num_blocks, num_threads>>>(gpu_args);

  gm.copy_to_host(fa->f, sizeof(field_t) * sp->g->nv);
}

std::map<int, double> energy_p_gpu_en;
void energy_p_gpu_stage_1(species_t *sp_list, interpolator_array_t *ia) {
  int num_threads, num_blocks, stride_size;

  species_t *sp;
  LIST_FOR_EACH(sp, sp_list) {
    if(sp->np <= 0) continue;
    energy_p_gpu_args gpu_args;

    gpu_args.qdt_2mc = (sp->q * sp->g->dt) / (2 * sp->m * sp->g->cvac);
    gpu_args.msp = sp->m;
    gpu_args.np = sp->np;

    energy_p_gpu_en[sp->id] = 0.0;

    gpu_args.p = (particle_t *)gm.map_to_device((host_pointer)sp->p, sizeof(particle_t) * sp->max_np);
    gpu_args.f = (interpolator_t *)gm.map_to_device((host_pointer)ia->i, sizeof(interpolator_t) * sp->g->nv);
    gpu_args.en = (double *)gm.copy_to_device((host_pointer)&energy_p_gpu_en[sp->id], sizeof(double));

    stride_size = DEFAULT_NP_STRIDE_SIZE;
    // num_threads = DEFAULT_NUM_THREAD;
    num_threads = 128;
    num_blocks = MATH_CEIL(sp->np, stride_size);
    gpu_args.stride_size = stride_size;
    energy_p_gpu<<<num_blocks, num_threads, 0, sp_streams[sp->id]>>>(gpu_args);
  }
}

double energy_p_gpu_stage_2(species_t *sp, interpolator_array_t *ia) {
  if(sp->np <= 0) return 0;
  cudaStreamSynchronize(sp_streams[sp->id]);
  gm.copy_to_host((host_pointer)&energy_p_gpu_en[sp->id], sizeof(double));

  double local = energy_p_gpu_en[sp->id];
  double global;
  mp_allsum_d(&local, &global, 1);

  return global * ((double)sp->g->cvac * (double)sp->g->cvac);
}

void copy_p_back(species_t *sp){
  gm.copy_to_host(sp->p, sp->max_np * sizeof(particle_t));

}

};  // namespace vpic_gpu
