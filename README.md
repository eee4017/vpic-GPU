# Vector Particle-In-Cell GPU version (vpic-GPU)

VPIC modfied version with GPU support for Student Cluster Competition 19, SC19

This repository is fork from https://github.com/lanl/vpic. Please refer to the original project for detail information.

We do NOT guarantee the correctness of the answer.

NTHU, Taiwan

## Compile

`arch/cuda` contains compile arguments designed for Nvidia&reg; Tesla&reg; V100.

```
module load cuda/10.0
mkdir build
cd build
../arch/cuda
make -j
```

## Test case: lpi_2d_F6_test

```
cd build/bin
mkdir lpi_2d_F6_test
cd lpi_2d_F6_test
cp ~/vpic-GPU/sample/lpi_2d_F6_test .
../vpic ./lpi_2d_F6_test
./lpi_2d_F6_test.Generic --tpp <thread num>
```

For multiple GPU environment, you should set CUDA_VISIBLE_DEVICES and excute with MPI, each device would be assigned to a MPI process. If you want to assign more process to a device, you could modify `mpiSetDevice()` inside `cuda_src/gpu.cu`.

```
mpirun -n <device num> ./lpi_2d_F6_test.Generic --tpp <thread num>
```

## Notification

1. The checkpointing should work correctly. However, we don't support checkpointing before the first timestep, you should disable checkpointing before the first timestep. (e.g. test case beam_plas)

2. Some features are not supported in this version:
    + emission  
    + injection

## Performance

`lpi_2d_F6_test` 2000 timesteps

- GPU version: `eee4017/vpic-GPU` `50835bb`
- CPU version: `lanl/vpic` `51d05ac`

Hardware Specification:
- GPU: Nvidia&reg; Tesla&reg; V100-PCIE-32GB * 4
- CPU: Intel&reg; Xeon&reg; Silver 4110 * 2 (16 core)

Sofware Specification:
- CUDA 10.0
- Intel&reg; Parallel Studio XE 2018 (include icc and intel mpi)

### Results


| version  | resources   | threads  |  mpi process | total time |
| -------- | --------    | -------- | -------- | -------- |
| CPU      |   16 cores  | 16       | 1        | 5:26.59  |
| CPU      |   16 cores  | 1        | 16       | 2:59.22  |
| GPU      |   1 device  | 4        | 1        | 3:08.45  |
| GPU      |   4 devices | 4        | 4        | 1:05.34  |


## Contributers of vpic-GPU

[@eee4017](https://github.com/eee4017) [@zlsh80826](https://github.com/zlsh80826) [@ttpssabc](https://github.com/ttpssabc)