#ifndef GLOBAL_H
#define GLOBAL_H

#include <iostream>
#include <fstream>
#include "macros.h"
#include "Kokkos_Core.hpp"


// define the used types and corresponding special functions as macros so they can be changed 
// (e.g. to benchmark double vs. float performance) 
// without refactoring the entire code
// also define macros for unrolled loops 
#ifndef USE_SINGLE_PRECISION
	#define USE_SINGLE_PRECISION true
#endif
#ifndef UNROLL_LOOP
	#define UNROLL_LOOP true
#endif

#if USE_SINGLE_PRECISION
	#define FLT float
	#define MFLOAT MPI_FLOAT
#else 
	#define FLT double
	#define MFLOAT MPI_DOUBLE
#endif

#define SIN sinf
#define SQRT sqrtf
#define INT int32_t
#define UINT unsigned long long


// define an accessor macro for fields so that their memory layout 
// can be changed without refactoring every single function
#if COALESCING
	#define VIEW(view, x, y, q) view(q, y, x)
#else
	#define VIEW(view, x, y, q) view(y, x, q)
#endif


// PARAMETERS

enum OUTPUT_TYPE{
	/// @brief no output
	NONE,
	/// @brief output the maximum velocity magnitude across all nodes as a scalar
	MAX_VEL,
	/// @brief output the magnitude of each node's velocity as a 2D matrix
	VEL_MAGS,
	/// @brief output the x- and y-components of each node's velocity as two 2D matrices, first x then y
	VEL_FIELD,
};
/// @brief the type of output printed to std::out when dumping data
inline OUTPUT_TYPE OUTPUT {OUTPUT_TYPE::NONE};

enum SCENE_TYPE{
	/// @brief simulate a shear-wave decay with x-velocities and periodic boundaries
	SHEAR_WAVE,
	/// @brief simulate a lid driven cavity with bounce-back solid walls
	LID_DRIVEN,
};
/// @brief the type of simulation to conduct (boundary and initial conditions)
inline SCENE_TYPE SCENE {SCENE_TYPE::SHEAR_WAVE};

/// @brief the stream to print output to. defaults to std::cout but may be overwritten with a file stream
inline std::ostream* OUT_STREAM{&std::cout};
/// @brief the file to write to, if specified and opened - otherwise, this is uninitialized! 
/// OUT_STREAM relies on this to have a static lifetime so the stream is not destructed before the program terminates.
inline std::ofstream OUT_FILE;

/// @brief Whether or not MPI is currently being used. If MPI is not used, use a more performant version for single GPUs.
inline bool USE_MPI {true};

/// @brief Whether to use a push-type streaming pattern instead of the default pull pattern.
inline bool PUSH {false};

/// @brief dump simulation results to std::out every so many timesteps
inline UINT OUT_EVERY_N{0};
/// @brief number of total time steps of the simulation
inline UINT STEPS{100};
/// @brief # grid points in y-direction
inline UINT NY{1024};
/// @brief # grid points in x-direction
inline UINT NX{1024};
/// @brief number of discrete velocity directions
const UINT Q{9};

/// @brief interpolation coefficient towards local equilibrium distribution
inline FLT OMEGA{1.7};
/// @brief initial density
inline FLT RHO_INIT{1.0};
/// @brief initial distribution values
inline FLT U_INIT{0.1};

/// @brief # Kokkos::MDRange tile group size x-direction
inline UINT TX{512};
/// @brief # Kokkos::MDRange tile group size y-direction
inline UINT TY{1};

// FIELD TYPES

/// @brief Field of density values (x,y)
using Den_t = Kokkos::View<FLT**, Kokkos::LayoutRight>;

/// @brief Field of velocity values (dir,x,y)
using Vel_t = Kokkos::View<FLT**[2], Kokkos::LayoutRight>;

/// @brief Host-side field of velocity values (dir,x,y)
using Vel_t_host = Kokkos::View<FLT**[2], Kokkos::LayoutRight, Kokkos::HostSpace::device_type, Kokkos::Experimental::DefaultViewHooks>;

/// @brief Field of distribution values
using Dst_t = Kokkos::View<FLT***, Kokkos::LayoutRight>; // Q Y X

/// @brief Field of coordinates of bounce-back, no-slip walls
using Bdy_t = Kokkos::View<UINT*[2], Kokkos::LayoutRight>;

/// @brief Field of packed, contiguous floats for MPI communication of halo regions
using Hlo_t = Kokkos::View<FLT*, Kokkos::LayoutRight>;

/// @brief A scalar, unsigned int
using DUI = Kokkos::View<UINT>;

/// @brief A scalar float
using DFL = Kokkos::View<FLT>;


/// @brief Utility function for creating a device-side accessible UINT view with value x
/// @param x the UINT to send to the device
/// @return the device-side view containing x
inline DUI create_device_uint(UINT x){
	DUI dx("device-side uint");
	auto host_dx = Kokkos::create_mirror_view(dx);
	host_dx() = x;
	Kokkos::deep_copy(dx, host_dx);
	return dx;
}

/// @brief Utility function for creating a device-side accessible FLT view with value x
/// @param x the FLT to send to the device
/// @return the device-side view containing x
inline DFL create_device_float(FLT x){
	DFL dx("device-side float");
	auto host_dx = Kokkos::create_mirror_view(dx);
	host_dx() = x;
	Kokkos::deep_copy(dx, host_dx);
	return dx;
}

#endif