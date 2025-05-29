# Name and CMAKE Version
cmake_minimum_required(VERSION 3.16)
project(LBM)

# Set the C++ Standard used to 17 for constexpr
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


# Build type: Release with debug symbols
if(NOT CMAKE_BUILD_TYPE)
  set(default_build_type "RelWithDebInfo")
  message(STATUS "Setting build type to '${default_build_type}' as none was specified.")
  set(CMAKE_BUILD_TYPE "${default_build_type}" CACHE STRING
    "Choose the type of build, options are: Debug, Release, RelWithDebInfo and MinSizeRel."
    FORCE)
endif()

# Try to locate the Kokkos dependency
set(Kokkos_COMMON_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/deps/Kokkos)

# Fetch Kokkos from github if not already present
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

target_link_libraries(main Kokkos::kokkos)

# Add CUDA-specific relaxed constexpr support
if(Kokkos_ENABLE_CUDA)
  target_compile_options(main PRIVATE 
    "$<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>"
  )
endif()
