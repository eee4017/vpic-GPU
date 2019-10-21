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

## run lpi_2d_F6_test

```
cd build/bin
mkdir lpi_2d_F6_test
cd lpi_2d_F6_test
cp ~/vpic-GPU/sample/lpi_2d_F6_test .
```

you can set the numstep in lpi_2d_F6_test to smaller number (e.g. num_step = 20)

```
../vpic ./lpi_2d_F6_test
./lpi_2d_F6_test.Generic --tpp <thread num>
```
