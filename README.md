# Vector Particle-In-Cell (VPIC) Project

VPIC modfied version for Student Cluster Competition 19, SC19

National Tsinghua University, Taiwan

## Compile

```
module load cuda/10.0
mkdir build
cd build
../arch/cuda
make -j
```

modified build/bin/vpic
1. change mpiicpc to `nvcc -ccbin=mpiicpc`
2. remove arguments which nvcc does not recognize -Wl....

compiler issue
if set vectorization in arch/cuda (e.g. `-DUSE_V16_AVX512`) to ON, nvcc cannot understand the flags inside header file.
