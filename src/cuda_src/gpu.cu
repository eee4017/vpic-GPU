#include "gpu.cuh"
#include "advance_p_gpu.cuh"
#include "gpu_util.cuh"

void advance_p_gpu_launcher(advance_p_pipeline_args_t *args){
    const int num_threads = 32;
    const int num_blocks = MATH_CEIL(args->np, 2048);
    advance_p_gpu<<<num_blocks, num_threads>>>(*args, 2048);
}
