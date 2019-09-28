#include "gpu.cuh"
#include "advance_p_gpu.cuh"
#include "sort_p_gpu.cuh"
#include "gpu_util.cuh"


namespace vpic_gpu{

    void advance_p_gpu_launcher(advance_p_pipeline_args_t *args){
        const int num_threads = 32;
        const int block_size = 2048;

        const int num_blocks = MATH_CEIL(args->np, block_size);
        advance_p_gpu_args gpu_args;

        gpu_args.qdt_2mc = args->qdt_2mc;
        gpu_args.cdt_dx = args->cdt_dx;
        gpu_args.cdt_dy = args->cdt_dy;
        gpu_args.cdt_dz = args->cdt_dz;
        gpu_args.qsp = args->cdt_dz;
        gpu_args.np = args->np;

        // gpu_particle_map.get_device_pointer(args->p0);
        

        advance_p_gpu<<<num_blocks, num_threads>>>(gpu_args);
    }

    void sort_p_gpu_launcher(species_t * sp){

      const int num_threads = 32;
      const int num_blocks = 512;

      particle_t * device_p;
      // should call get_device_pointer(device_p) here

      //******************************************************
      //*****modified from cub's device_radixsort example*****
      //******************************************************

      int num_items = sp->np;

      // DoubleBuffer(index/particle) for sorting
      DoubleBuffer<int32_t>      d_keys;
      DoubleBuffer<particle_t>   d_values;

      CubDebugExit(g_allocator.DeviceAllocate((void**)&d_keys.d_buffers[0], sizeof(int32_t) * num_items));
      CubDebugExit(g_allocator.DeviceAllocate((void**)&d_keys.d_buffers[1], sizeof(int32_t) * num_items));
      CubDebugExit(g_allocator.DeviceAllocate((void**)&d_values.d_buffers[0], sizeof(particle_t) * num_items));
      CubDebugExit(g_allocator.DeviceAllocate((void**)&d_values.d_buffers[1], sizeof(particle_t) * num_items));

      // Allocate temporary storage
      size_t  temp_storage_bytes  = 0;
      void    *d_temp_storage     = NULL;

      CubDebugExit(DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, d_keys, d_values, num_items));
      CubDebugExit(g_allocator.DeviceAllocate(&d_temp_storage, temp_storage_bytes));

      // Initialize device arrays
      copy_particle_index<<<num_blocks, num_threads>>>(d_keys.d_buffers[0]  , device_p, num_items);
      copy_particle      <<<num_blocks, num_threads>>>(d_values.d_buffers[0], device_p, num_items);

      // Run
      CubDebugExit(DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, d_keys, d_values, num_items));

      // Copy particles back
      copy_particle<<<num_blocks, num_threads>>>(device_p, d_values.Current(), num_items);

      // Cleanup
      if (d_keys.d_buffers[0]) CubDebugExit(g_allocator.DeviceFree(d_keys.d_buffers[0]));
      if (d_keys.d_buffers[1]) CubDebugExit(g_allocator.DeviceFree(d_keys.d_buffers[1]));
      if (d_values.d_buffers[0]) CubDebugExit(g_allocator.DeviceFree(d_values.d_buffers[0]));
      if (d_values.d_buffers[1]) CubDebugExit(g_allocator.DeviceFree(d_values.d_buffers[1]));
      if (d_temp_storage) CubDebugExit(g_allocator.DeviceFree(d_temp_storage));
    }
};
