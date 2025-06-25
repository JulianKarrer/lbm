# Name and CMAKE Version
cmake_minimum_required(VERSION 3.16)
project(LBM)

# Set the C++ Standard used to 17 for constexpr
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(default_build_type "Release")
  message(STATUS "Setting build type to '${default_build_type}' as none was specified.")
  set(CMAKE_BUILD_TYPE "${default_build_type}" CACHE STRING
    "Choose the type of build, options are: Debug, Release, RelWithDebInfo and MinSizeRel."
    FORCE)
endif()

# set compiler optimization flags  
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -DNDEBUG" CACHE STRING "Flags for Release build" FORCE)
set(CMAKE_CUDA_FLAGS_RELEASE "-O3 --use_fast_math -DNDEBUG" CACHE STRING "Flags for CUDA Release build" FORCE)

# set Kokkos flags
set(Kokkos_ENABLE_SERIAL      ON  CACHE BOOL "")
set(Kokkos_ENABLE_OPENMP      OFF CACHE BOOL "")
set(Kokkos_ENABLE_THREADS     OFF CACHE BOOL "")
set(Kokkos_ENABLE_CUDA        ON  CACHE BOOL "")
set(Kokkos_ENABLE_PROFILING   OFF CACHE BOOL "")
set(Kokkos_ENABLE_DEBUG       OFF CACHE BOOL "")
set(Kokkos_ENABLE_TESTS       OFF CACHE BOOL "")
set(Kokkos_ENABLE_EXAMPLES    OFF CACHE BOOL "")
set(Kokkos_ENABLE_KOKKOSKERNELS ON CACHE BOOL "")
set(Kokkos_ARCH_NATIVE        ON  CACHE BOOL "")  # auto‐detect
set(Kokkos_ARCH_AMPERE_80     ON  CACHE BOOL "") 
set(DKokkos_ENABLE_CUDA_LAMBDA ON  CACHE BOOL "")

set(Kokkos_COMMON_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/deps/Kokkos)
find_package(Kokkos CONFIG)
if(Kokkos_FOUND)
  message(STATUS "Found Kokkos: ${Kokkos_DIR} (version \"${Kokkos_VERSION}\")")
else()
  if(EXISTS ${Kokkos_COMMON_SOURCE_DIR})
    add_subdirectory(${Kokkos_COMMON_SOURCE_DIR} Kokkos)
  else()
    include(FetchContent)
    FetchContent_Declare(
      Kokkos
      GIT_REPOSITORY https://github.com/kokkos/kokkos.git
      GIT_TAG        4.5.01
      SOURCE_DIR ${Kokkos_COMMON_SOURCE_DIR}
    )
    FetchContent_MakeAvailable(Kokkos)
  endif()
endif()

# Add argparse
include_directories(deps)

# Add the main executable and link it against Kokkos
add_executable(
  main 
  src/main.cpp
  deps/argparse/argparse.hpp
)

# include MPI support
find_package(MPI REQUIRED)

# add an option to pass an optimization level for benchmarks
option(OPT_LEVEL "Optimization setting to use in the code. Used for benchmarking." 0)
if (OPT_LEVEL)
  message(STATUS "Using optimization level ${OPT_LEVEL}.")
  target_compile_definitions(main PRIVATE OPT=${OPT_LEVEL})
endif()

target_link_libraries(main Kokkos::kokkos MPI::MPI_CXX)
