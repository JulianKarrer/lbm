# D2Q9 LATTICE BOLTZMANN
Using Kokkos with CUDA/HIP/SYCL for GPU-parallelization of a Lattice Boltzmann solver and comparing performance to a Taichi implementation. Made for the summer 2025 course at the University of Freiburg [High-performance computing: Distributed-memory parallelization on GPUs and accelerators](https://pastewka.github.io/Accelerators/)

### Build
Setup
```bash
cmake -S . -B build
```
For the A100 nodes on the BWUniCluster use `cmake -S . -B build -DKokkos_ARCH_AMPERE80=ON` to avoid missing auto-detection of the target architecture. For H100, try `-DKokkos_ARCH_HOPPER90=ON`. On NEMO2 with MI300A, `CMAKE_PREFIX_PATH=/opt/rocm CC=/opt/rocm/bin/amdclang CXX=/opt/rocm/bin/amdclang++ cmake -S . -B build -DKokkos_ARCH_AMD_GFX942_APU=ON -DKokkos_ENABLE_HIP=ON` might be appropriate.

Compile 
```bash
cmake --build build
```

Adjust `Kokkos_ENABLE_CUDA` etc. in `CMakeLists.txt` for HIP/SYCL/...


## Command Line Options

```
Usage: lbm [--help] [--version] --x-grid-points VAR --y-grid-points VAR --output-frequency VAR --steps VAR --omega VAR --density VAR --initial-velocity VAR --x-grid-points VAR --y-grid-points VAR [--push] [[--output-max-vel]|[--output-velocity]|[--output-velocity-field]] [--file VAR] [[--shear-wave-decay]|[--lid-driven-cavity]]

Optional arguments:
  -h, --help                     shows help message and exits
  -v, --version                  prints version information and exits
  -nx, --x-grid-points           Specify the number of grid points in the x-direction [nargs=0..1] [default: 1024]
  -ny, --y-grid-points           Specify the number of grid points in the y-direction [nargs=0..1] [default: 1024]
  -of, --output-frequency        Specify how frequently (as an integer number of timesteps) simulation results should be output. 0 means no output. [nargs=0..1] [default: 0]
  -s, --steps                    Specify the total number of simulation time steps to run [nargs=0..1] [default: 50000]
  -w, --omega                    Specify the relaxation coefficient omega, which should obey be a float in (0;2) [nargs=0..1] [default: 0.5]
  -rho, --density                Specify the initial density at each fluid node [nargs=0..1] [default: 1]
  -u, --initial-velocity         Specify the initial velocity. For the shear wave decay, this is the velocity amplitude. For the lid-driven cavity, this is the horizontal velocity of the top lid wall. [nargs=0..1] [default: 0.1]
  -tx, --x-grid-points           Specify the kernel tiling size in x-direction for performance optimization [nargs=0..1] [default: 512]
  -ty, --y-grid-points           Specify the kernel tiling size in y-direction for performance optimization [nargs=0..1] [default: 1]
  -push, --push                  Specify whether to prefer a push-type streaming pattern over pull-type. Pulling is used by default.
  -omv, --output-max-vel         If specified, the program outputs the maximum velocity magnitude as a comma-seperated list of floats with a trailing comma
  -ov, --output-velocity         If specified, the program outputs the velocity magnitude at each node as #-seperated CSV tables of NY columns and NX rows
  -ovf, --output-velocity-field  If specified, the program outputs the components of the velocity field at each node as #-seperated CSV tables of NY columns and NX rows, first all x-components, then all y-components
  -f, --file                     Specify a filename to write to instead of printing output to std::cout
  -sw, --shear-wave-decay        Simulate a shear-wave decay with velocities in x-direction and periodic boundaries. This is the default.
  -ld, --lid-driven-cavity       Simulate a lid driven cavity with bounce-back solid walls and a moving wall at the top, where the fluid is initially at rest.
```

## Taichi version

To run the LBM solver in the `taichi` directory, Python 3.10 (see `.python-version`) and `python3 -m pip install taichi numpy matplotlib` are required
