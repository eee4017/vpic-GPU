#include "gpu.cuh"
#include "advance_p_gpu.cuh"
#include "energy_p_gpu.cuh"
#include "accumulate_rho_p_gpu.cuh"
#include "sort_p_gpu.cuh"
#include "backfill_gpu.cuh"
#include "gpu_util.cuh"
#include <CppToolkit/color.h>
#include <cub/cub.cuh>
#include <cub/util_allocator.cuh>

#include <map>

namespace vpic_gpu{
  	gpu_memory_allocator gm;

    void sort_p_gpu_launcher(species_t * sp){

      const int num_threads = 32;
      const int num_blocks = 512;

      particle_t * device_p = (particle_t *)gm.map_to_device(sp->p, sizeof(particle_t) * sp->np);

      //******************************************************
      //*****modified from cub's device_radixsort example*****
      //******************************************************

      int num_items = sp->np;

      // DoubleBuffer(index/particle) for sorting
      cub::DoubleBuffer<int32_t>      d_keys;
      cub::DoubleBuffer<particle_t>   d_values;
      cub::CachingDeviceAllocator  g_allocator(true); 

      CubDebugExit(g_allocator.DeviceAllocate((void**)&d_keys.d_buffers[0], sizeof(int32_t) * num_items));
      CubDebugExit(g_allocator.DeviceAllocate((void**)&d_keys.d_buffers[1], sizeof(int32_t) * num_items));
      CubDebugExit(g_allocator.DeviceAllocate((void**)&d_values.d_buffers[0], sizeof(particle_t) * num_items));
      CubDebugExit(g_allocator.DeviceAllocate((void**)&d_values.d_buffers[1], sizeof(particle_t) * num_items));

      // Allocate temporary storage
      size_t  temp_storage_bytes  = 0;
      void    *d_temp_storage     = NULL;

      CubDebugExit(cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, d_keys, d_values, num_items));
      CubDebugExit(g_allocator.DeviceAllocate(&d_temp_storage, temp_storage_bytes));

      // Initialize device arrays
      copy_particle_index<<<num_blocks, num_threads>>>(d_keys.d_buffers[0]  , device_p, num_items);
      copy_particle      <<<num_blocks, num_threads>>>(d_values.d_buffers[0], device_p, num_items);

      // Run
      CubDebugExit(cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, d_keys, d_values, num_items));

      // Copy particles back
      copy_particle<<<num_blocks, num_threads>>>(device_p, d_values.Current(), num_items);

      // Cleanup
      if (d_keys.d_buffers[0]) CubDebugExit(g_allocator.DeviceFree(d_keys.d_buffers[0]));
      if (d_keys.d_buffers[1]) CubDebugExit(g_allocator.DeviceFree(d_keys.d_buffers[1]));
      if (d_values.d_buffers[0]) CubDebugExit(g_allocator.DeviceFree(d_values.d_buffers[0]));
      if (d_values.d_buffers[1]) CubDebugExit(g_allocator.DeviceFree(d_values.d_buffers[1]));
      if (d_temp_storage) CubDebugExit(g_allocator.DeviceFree(d_temp_storage));
    }

    void mpiSetDevice(int rank){
      // MY_MESSAGE( ("setting devices@%d", rank) );
      cudaSetDevice(rank);
    }

    void advance_p_gpu_launcher(advance_p_pipeline_args_t *args, species_t * sp){
        int num_threads = 32;
        int block_size = 2048;
        // int block_size = 512;
        // int num_threads = block_size;

        int num_blocks = MATH_CEIL(args->np, block_size);
        advance_p_gpu_args gpu_args;

        gpu_args.qdt_2mc = args->qdt_2mc;
        gpu_args.cdt_dx = args->cdt_dx;
        gpu_args.cdt_dy = args->cdt_dy;
        gpu_args.cdt_dz = args->cdt_dz;
        gpu_args.qsp = args->qsp;
        gpu_args.np = args->np;
        
        gpu_args.block_size = block_size;
        gpu_args.g_rangel = args->g->rangel;
        gpu_args.g_rangeh = args->g->rangeh;
        
        sp->nm = 0;
        gpu_args.nm = (int *)gm.copy_to_device(&sp->nm, sizeof(int));
        gpu_args.pm_array = (particle_mover_t *)gm.map_to_device(args->pm, sizeof(particle_mover_t) * args->max_nm);
        gpu_args.temp_pm_array = (particle_mover_t *)gm.map_to_device(args->pm, sizeof(particle_mover_t) * args->max_nm);

        gpu_args.p0 = (particle_t *)gm.map_to_device(args->p0, sizeof(particle_t) * args->np);
        gpu_args.a0 = (accumulator_t *)gm.map_to_device(args->a0, sizeof(accumulator_t) * args->g->nv);
        cudaMemset(gpu_args.a0, 0, sizeof(accumulator_t) * args->g->nv); // clear accumulator array
        gpu_args.f0 = (interpolator_t *)gm.copy_to_device((host_pointer)args->f0, sizeof(interpolator_t) * args->g->nv);
        gpu_args.g_neighbor = (int64_t *)gm.map_to_device(args->g->neighbor, sizeof(int64_t) * args->g->nv * 6);
        
        cudaTimer advance_timer;
        advance_timer.start();
        advance_p_gpu<<<num_blocks, num_threads>>>(gpu_args);
        gpuErrchk( cudaPeekAtLastError() );
        advance_timer.end();
        // advance_timer.printTime("advance_timer");

        gm.copy_to_host(&sp->nm, sizeof(int));
        // MY_MESSAGE( ("gpu sp->nm: %d", sp->nm) );
        //********************HANDLE PM**********************//
        int temp_nm = sp->nm;
        block_size = 256;
        num_threads = min(temp_nm, block_size);
        num_blocks = MATH_CEIL(temp_nm, block_size);
        gpu_args.block_size = block_size;

        sp->nm = 0;
        gpu_args.nm = (int *)gm.copy_to_device(&sp->nm, sizeof(int));
        
        handle_particle_movers<<<num_blocks, num_threads>>>(gpu_args, temp_nm);
        gpuErrchk( cudaPeekAtLastError() );

        gm.copy_to_host(args->a0, sizeof(accumulator_t) * args->g->nv);
        gm.copy_to_host(&sp->nm, sizeof(int));
    }


    void boundary_p_get_p_pm(particle_t* p0,  particle_mover_t* pm, species_t* sp){

      particle_t * device_p = (particle_t *)gm.map_to_device(sp->p, sizeof(particle_t) * sp->np);
      particle_mover_t * device_pm = (particle_mover_t *)gm.map_to_device(sp->pm, sizeof(particle_mover_t) * sp->nm);

      int np = sp->np;
      int nm = sp->nm;

      if(nm <= 0){
        return;
      }

      //allocate device p0/pm
      particle_t * d_p0;
      particle_t * device_particle_temp;
      int *device_particle_counter;
      gpuErrchk( cudaMalloc(&d_p0, nm * sizeof(particle_t)) );
      gpuErrchk( cudaMalloc(&device_particle_temp,  nm * sizeof(particle_t)) );

      gpuErrchk( cudaMalloc(&device_particle_counter, sizeof(int)) );
      gpuErrchk( cudaMemset(device_particle_counter, 0, sizeof(int)) );
      
      //find all device p0/pm with back-filling
      int block_size = 256;
      const int num_threads = min(nm, block_size);
      const int num_blocks = MATH_CEIL(nm, block_size);
      
      back_fill_read<<<num_blocks, num_threads>>>(device_p, device_pm, 
                                                  device_particle_temp, device_particle_counter,
                                                  d_p0, np, nm, block_size);
      gpuErrchk( cudaPeekAtLastError() );
      back_fill_write<<<num_blocks, num_threads>>>(device_p, device_pm, 
                                                  device_particle_temp, device_particle_counter,
                                                  d_p0, np, nm, block_size);

      gpuErrchk( cudaPeekAtLastError() );


      //transfer device to host
      gpuErrchk( cudaMemcpy(p0, d_p0, nm * sizeof(particle_t), cudaMemcpyDeviceToHost));
      gpuErrchk( cudaMemcpy(pm, device_pm, nm * sizeof(particle_mover_t), cudaMemcpyDeviceToHost));

      //free device p0/pm
      gpuErrchk( cudaFree(d_p0) );
      gpuErrchk( cudaFree(device_particle_temp) );
      gpuErrchk( cudaFree(device_particle_counter) );
    }

    // __global__
    // void check_p_idx(particle_t *device_p, int np, int pi_cnt){
    //   particle p = device_p[blockIdx.x * blockDim.x + threadIdx.x];
    //   if( p.i < 0 ){
    //    printf ("check_p_idx: ERROR idx %d@%d int last %d\n", p.i, np - blockIdx.x * blockDim.x + threadIdx.x, pi_cnt) ;
    //   }
    // }

    void append_p_and_pm(particle_t * temp_p, particle_mover_t *temp_pm,
                         int pi_cnt, int pm_cnt, species_t * sp) {
      particle_t * device_p = (particle_t *)gm.map_to_device(sp->p, sizeof(particle_t) * sp->np);
      particle_mover_t * device_pm = (particle_mover_t *)gm.map_to_device(sp->pm, sizeof(particle_mover_t) * sp->nm);
      
      gpuErrchk( cudaMemcpy(device_p + sp->np - pi_cnt, temp_p, pi_cnt * sizeof(particle_t),       cudaMemcpyHostToDevice));
      gpuErrchk( cudaMemcpy(device_pm + sp->nm - pm_cnt, temp_pm, pm_cnt * sizeof(particle_mover_t), cudaMemcpyHostToDevice));
      
      // const int num_threads = 512;
      // const int num_blocks = MATH_CEIL(sp->np, num_threads);
      // check_p_idx<<<num_blocks, num_threads>>>(device_p, sp->np, pi_cnt);
    }

    void accumulate_rho_p_gpu_launcher(field_array_t* fa, const species_t* sp){
      const int block_size = 2048;
      const int num_threads = 32;
      const int num_blocks = MATH_CEIL(sp->np, block_size);

      accumulate_rho_p_gpu_args gpu_args;

      gpu_args.q_8V = sp->q * sp->g->r8V;
      gpu_args.np = sp->np;
      gpu_args.sy = sp->g->sy;
      gpu_args.sz = sp->g->sz;

      gpu_args.block_size = block_size;
      gpu_args.p = (particle_t *)gm.map_to_device(sp->p, sizeof(particle_t) * sp->np);
      gpu_args.f = (field_t *)gm.copy_to_device(fa->f, sizeof(field_t) * sp->g->nv);
      
      // cudaTimer accumulate_rho_p_gpu_timer;      
      // accumulate_rho_p_gpu_timer.start();
      accumulate_rho_p_gpu<<<num_blocks, num_threads>>>(gpu_args);
      gpuErrchk( cudaPeekAtLastError() );
      // accumulate_rho_p_gpu_timer.end();
      // accumulate_rho_p_gpu_timer.printTime("accumulate_rho_p_gpu");
      
      gm.copy_to_host(fa->f, sizeof(field_t) * sp->g->nv);
    }

    std::map<int, double> energy_p_gpu_en;
    void energy_p_gpu_launcher_1st(species_t * sp_list, interpolator_array_t *ia){
      const int block_size = 2048;
      const int num_threads = 32;
      
      species_t *sp;
    LIST_FOR_EACH( sp, sp_list ) {
      energy_p_gpu_args gpu_args;
      int num_blocks = MATH_CEIL(sp->np, block_size);

      gpu_args.qdt_2mc = (sp->q*sp->g->dt)/(2*sp->m*sp->g->cvac);
      gpu_args.msp = sp->m;
      gpu_args.np = sp->np;
      
      gpu_args.block_size = block_size;
      energy_p_gpu_en[sp->id] = 0.0;
      gpu_args.p = (particle_t *)gm.map_to_device((host_pointer)sp->p, sizeof(particle_t) * sp->np);
      gpu_args.f = (interpolator_t *)gm.map_to_device((host_pointer)ia->i, sizeof(interpolator_t) * sp->g->nv);
      gpu_args.en = (double *)gm.copy_to_device((host_pointer)&energy_p_gpu_en[sp->id], sizeof(double));

      energy_p_gpu<<<num_blocks, num_threads>>>(gpu_args);
    }

    }

    double energy_p_gpu_launcher_2nd(species_t * sp, interpolator_array_t *ia){
      gpuErrchk( cudaPeekAtLastError() );
      gm.copy_to_host((host_pointer)&energy_p_gpu_en[sp->id], sizeof(double));
      
      double local = energy_p_gpu_en[sp->id];
      double global;
      mp_allsum_d( &local, &global, 1 );
  
      return global * ( ( double ) sp->g->cvac * ( double ) sp->g->cvac );
    }

};
