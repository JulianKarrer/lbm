# D2Q9 LATTICE BOLTZMANN
Using Kokkos with CUDA/HIP/SYCL for massive parallelization

### Build
Setup
```
cmake -S . -B build -DKokkos_ENABLE_CUDA=ON -DKokkos_ENABLE_CUDA_LAMBDA=ON
```
Compile 
```
cmake --build build
```