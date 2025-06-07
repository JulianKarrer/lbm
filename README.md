# D2Q9 LATTICE BOLTZMANN
Using Kokkos with CUDA/HIP/SYCL for GPU-parallelization of a Lattice Boltzmann solver and comparing performance to a Taichi implementation. Made for the summer 2025 course at the University of Freiburg [High-performance computing: Distributed-memory parallelization on GPUs and accelerators](https://pastewka.github.io/Accelerators/)

### Build
Setup
```
cmake -S . -B build
```
For the A100 nodes on the BWUniCluster use `cmake -S . -B build -DKokkos_ARCH_AMPERE80=ON` to avoid missing auto-detection of the target architecture. 

Compile 
```
cmake --build build
```

Adjust `Kokkos_ENABLE_CUDA` etc. in `CMakeLists.txt` for HIP/SYCL/...


## Taichi version

To run the LBM solver in the `taichi` directory, Python 3.10 (see `.python-version`) and `python3 -m pip install taichi numpy matplotlib` are required
