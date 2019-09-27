#ifndef __NTHU_GPU_H__
#define __NTHU_GPU_H__

#define IN_spa
#include "src/species_advance/standard/pipeline/spa_private.h"

void advance_p_gpu_launcher(advance_p_pipeline_args_t *);

#endif