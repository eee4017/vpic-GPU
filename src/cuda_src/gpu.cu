#include "gpu.cuh"
#include "advance_p_gpu.cuh"
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
};
