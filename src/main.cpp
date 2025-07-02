#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <chrono>
#include "Kokkos_Core.hpp"
#include <mpi.h>
#include "macros.h"
#include "global.h"
#include "io.h"
#include "init.h"


// FUNCTIONS

KOKKOS_INLINE_FUNCTION
void pull_periodic(const UINT x, const UINT y, const UINT xl, const UINT xr, const UINT yd, const UINT yu, Dst_t const& f, Dst_t const& buf, FLT OMEGA){
	// ###### STREAM ##################################################
	// here, stream from neighbours before colliding by "pulling" in surrounding values
	// | 6   2   5 |
	// |   \ | /   |
	// | 3 - 0 - 1 |
	// |   / | \   |
	// | 7   4   8 |
	const FLT f_7 { VIEW(f,xr,yu, 7) };
	const FLT f_4 { VIEW(f, x,yu, 4) };
	const FLT f_8 { VIEW(f,xl,yu, 8) };
	const FLT f_3 { VIEW(f,xr, y, 3) };
	const FLT f_0 { VIEW(f, x, y, 0) };
	const FLT f_1 { VIEW(f,xl, y, 1) };
	const FLT f_6 { VIEW(f,xr,yd, 6) };
	const FLT f_2 { VIEW(f, x,yd, 2) };
	const FLT f_5 { VIEW(f,xl,yd, 5) };

	// ###### COMPUTE DENSITIES #######################################
	// collect density contributions from all directions
	const FLT rho {f_0 + f_1 + f_2 + f_3 + f_4 + f_5 + f_6 + f_7 + f_8};
	constexpr FLT f1 {1.0};
	const FLT rho_inv {f1/rho};

	// ###### COMPUTE VELOCITIES ######################################
	// contributions to x-velocities come from channels 1, 3, 5, 6, 7, 8
	/// 3,6,7 get a minus sign, 0,2,4 don't contribute
	const FLT ux { (f_1 - f_3 + f_5 - f_6 - f_7 + f_8) * rho_inv};
	// contributions to y-velocities come from channels 2, 4, 5, 6, 7, 8
	/// 4,7,8 get a minus sign, 0,1,3 don't contribute
	const FLT uy {(f_2 - f_4 + f_5 + f_6 - f_7 - f_8) * rho_inv};

	// define constantexpr with macro FLOAT beforehand to enable optimization 
	// while avoiding narrowing conversion - no matter if FLOAT is float or double
	constexpr FLT f45 {4.5};
	constexpr FLT f15 {1.5};
	constexpr FLT f49 {4./9.};
	constexpr FLT f19 {1./9.};
	constexpr FLT f136 {1./36.};


	// ###### COMPUTE EQUILIBIRUM DISTRIBUITON AND STREAM #############
	// What the compiler ought to already do:
	// - factor out common subexpressions
	// - avoid pointless arithmetic like *1, *0, use - instead of *(-1)
	// - use conditionals instead of %
	// Other optimizations:
	// - reorder writes so they access subsequent memory
	// - interleave required common subexpressions and writes
	const FLT w_1_36 {rho * f136};
	const FLT uxpuy {ux + uy};
	const FLT uxpuy_2 {f45 * uxpuy * uxpuy};
	const FLT u_2_times_3_2 {f15 * (ux * ux + uy * uy)};
	VIEW(buf, x, y, 7) = f_7 + OMEGA * ((w_1_36 * (1- 3*uxpuy+ uxpuy_2- u_2_times_3_2)) - f_7);
	const FLT uy_2 {f45 * uy * uy};
	const FLT w_1_9  {rho * f19};
	VIEW(buf, x, y, 4) = f_4 + OMEGA * ((w_1_9 * (1- 3*uy+ uy_2- u_2_times_3_2)) - f_4);
	const FLT uymux {uy - ux};
	const FLT uymux_2 {f45 * uymux * uymux};
	VIEW(buf, x, y, 8) = f_8 + OMEGA * ((w_1_36 * (1- 3*uymux+ uymux_2- u_2_times_3_2)) - f_8);
	const FLT ux_2 {f45 * ux * ux};
	VIEW(buf, x, y, 3) = f_3 + OMEGA * ((w_1_9 * (1- 3*ux+ ux_2- u_2_times_3_2)) - f_3);
	const FLT w_4_9  {rho * f49};
	VIEW(buf, x, y, 0) = f_0 + OMEGA * ((w_4_9 * (1- u_2_times_3_2)) - f_0);
	VIEW(buf, x, y, 1) = f_1 + OMEGA * ((w_1_9 * (1+ 3*ux+ ux_2- u_2_times_3_2)) - f_1);
	VIEW(buf, x, y, 6) = f_6 + OMEGA * ((w_1_36 * (1+ 3*uymux+ uymux_2- u_2_times_3_2)) - f_6);
	VIEW(buf, x, y, 2) = f_2 + OMEGA * ((w_1_9 * (1+ 3*uy+ uy_2- u_2_times_3_2)) - f_2);
	VIEW(buf, x, y, 5) = f_5 + OMEGA * ((w_1_36 * (1+ 3*uxpuy+ uxpuy_2- u_2_times_3_2)) - f_5);
}


KOKKOS_INLINE_FUNCTION
void pull_periodic_for(const UINT x, const UINT y, const UINT xl, const UINT xr, const UINT yd, const UINT yu, Dst_t const& f_i, Dst_t const& buf, FLT OMEGA){
	const FLT f[9]{ 
		VIEW(f_i, x, y, 0),
		VIEW(f_i,xl, y, 1),
		VIEW(f_i, x,yd, 2),
		VIEW(f_i,xr, y, 3),
		VIEW(f_i, x,yu, 4),
		VIEW(f_i,xl,yd, 5),
		VIEW(f_i,xr,yd, 6),
		VIEW(f_i,xr,yu, 7),
		VIEW(f_i,xl,yu, 8),
	};

	// ###### COMPUTE DENSITIES #######################################
	FLT rho {0.};
	for (UINT i{0}; i<9; ++i){
		rho += f[i];
	}
	constexpr FLT f1 {1.0};
	const FLT rho_inv {f1/rho};

	// ###### COMPUTE VELOCITIES ######################################
	constexpr FLT cx[9] { 0, 1, 0,-1, 0, 1,-1,-1, 1};
	constexpr FLT cy[9] { 0, 0, 1, 0,-1, 1, 1,-1,-1};
	FLT ux{0.};
	FLT uy{0.};
	for (UINT i{0}; i<9; ++i){
		ux += f[i]*cx[i];
		uy += f[i]*cy[i];
	}
	ux *= rho_inv;
	uy *= rho_inv;

	// ###### COMPUTE EQUILIBIRUM DISTRIBUITON WRITE TO BUFFER ########
	constexpr FLT w[9] = {4./9.,1./9.,1./9.,1./9.,1./9.,1./36.,1./36.,1./36.,1./36.,};
	for (UINT i{0}; i<9; ++i){
		const FLT cx_ux{cx[i] * ux};
		const FLT cy_uy{cy[i] * uy};
		const FLT c_i_u_r = cx_ux + cy_uy;
		const FLT u_2 = ux * ux + uy * uy;
		const FLT f_eq{ w[i] * rho * (((FLT)1)+ ((FLT)3)*c_i_u_r+ ((FLT)4.5) * (c_i_u_r * c_i_u_r)- ((FLT)1.5) * u_2)};
		VIEW(buf, x, y, i) = f[i] + OMEGA * (f_eq - f[i]);
	}
}


KOKKOS_INLINE_FUNCTION
void push_periodic(const UINT x, const UINT y, const UINT xl, const UINT xr, const UINT yd, const UINT yu, Dst_t const& f, Dst_t const& buf, FLT OMEGA){
	// ###### READ VALUES #############################################
	const FLT f_7 { VIEW(f, x, y, 7) };
	const FLT f_4 { VIEW(f, x, y, 4) };
	const FLT f_8 { VIEW(f, x, y, 8) };
	const FLT f_3 { VIEW(f, x, y, 3) };
	const FLT f_0 { VIEW(f, x, y, 0) };
	const FLT f_1 { VIEW(f, x, y, 1) };
	const FLT f_6 { VIEW(f, x, y, 6) };
	const FLT f_2 { VIEW(f, x, y, 2) };
	const FLT f_5 { VIEW(f, x, y, 5) };
	
	// ###### COMPUTE DENSITIES #######################################
	// collect density contributions from all directions
	const FLT rho {f_0 + f_1 + f_2 + f_3 + f_4 + f_5 + f_6 + f_7 + f_8};
	constexpr FLT f1 {1.0};
	const FLT rho_inv {f1/rho};

	// ###### COMPUTE VELOCITIES ######################################
	// contributions to x-velocities come from channels 1, 3, 5, 6, 7, 8
	/// 3,6,7 get a minus sign, 0,2,4 don't contribute
	const FLT ux { (f_1 - f_3 + f_5 - f_6 - f_7 + f_8) * rho_inv};
	// contributions to y-velocities come from channels 2, 4, 5, 6, 7, 8
	/// 4,7,8 get a minus sign, 0,1,3 don't contribute
	const FLT uy {(f_2 - f_4 + f_5 + f_6 - f_7 - f_8) * rho_inv};

	// Common subexpressions and constexprs of FLOAT type
	constexpr FLT f45 {4.5};
	constexpr FLT f15 {1.5};
	constexpr FLT f49 {4./9.};
	constexpr FLT f19 {1./9.};
	constexpr FLT f136 {1./36.};
	const FLT ux_2 {f45 * ux * ux};
	const FLT uy_2 {f45 * uy * uy};
		// cross terms
	const FLT uymux {uy - ux};
	const FLT uymux_2 {f45 * uymux * uymux};
	const FLT uxpuy {ux + uy};
	const FLT uxpuy_2 {f45 * uxpuy * uxpuy};
	const FLT u_2_times_3_2 {f15 * (ux * ux + uy * uy)};
	// weights and density
	const FLT w_4_9  {rho * f49};
	const FLT w_1_9  {rho * f19};
	const FLT w_1_36 {rho * f136};

	// order writes such that changes along the fastest varying index into memory are subsequent
	// | 6   2   5 |
	// |   \ | /   |
	// | 3 - 0 - 1 |
	// |   / | \   |
	// | 7   4   8 |
	VIEW(buf, xl, yd, 7) = f_7 + OMEGA * ((w_1_36 * (1- 3*uxpuy+ uxpuy_2- u_2_times_3_2)) - f_7);
	VIEW(buf, x , yd, 4) = f_4 + OMEGA * ((w_1_9 * (1- 3*uy+ uy_2- u_2_times_3_2)) - f_4);
	VIEW(buf, xr, yd, 8) = f_8 + OMEGA * ((w_1_36 * (1- 3*uymux+ uymux_2- u_2_times_3_2)) - f_8);

	VIEW(buf, xl, y , 3) = f_3 + OMEGA * ((w_1_9 * (1- 3*ux+ ux_2- u_2_times_3_2)) - f_3);
	VIEW(buf, x , y , 0) = f_0 + OMEGA * ((w_4_9 * (1- u_2_times_3_2)) - f_0);
	VIEW(buf, xr, y , 1) = f_1 + OMEGA * ((w_1_9 * (1+ 3*ux+ ux_2- u_2_times_3_2)) - f_1);

	VIEW(buf, xl, yu, 6) = f_6 + OMEGA * ((w_1_36 * (1+ 3*uymux+ uymux_2- u_2_times_3_2)) - f_6);
	VIEW(buf, x , yu, 2) = f_2 + OMEGA * ((w_1_9 * (1+ 3*uy+ uy_2- u_2_times_3_2)) - f_2);
	VIEW(buf, xr, yu, 5) = f_5 + OMEGA * ((w_1_36 * (1+ 3*uxpuy+ uxpuy_2- u_2_times_3_2)) - f_5);
}


KOKKOS_INLINE_FUNCTION
void push_periodic_for(const UINT x, const UINT y, const UINT xl, const UINT xr, const UINT yd, const UINT yu, Dst_t const& f_i, Dst_t const& buf, FLT OMEGA){
	// ###### READ VALUES #############################################
	const FLT f[9]{ 
		VIEW(f_i, x, y, 0),
		VIEW(f_i, x, y, 1),
		VIEW(f_i, x, y, 2),
		VIEW(f_i, x, y, 3),
		VIEW(f_i, x, y, 4),
		VIEW(f_i, x, y, 5),
		VIEW(f_i, x, y, 6),
		VIEW(f_i, x, y, 7),
		VIEW(f_i, x, y, 8),
	};
	
	// ###### COMPUTE DENSITIES #######################################
	FLT rho {0.};
	for (UINT i{0}; i<9; ++i){
		rho += f[i];
	}
	constexpr FLT f1 {1.0};
	const FLT rho_inv {f1/rho};

	// ###### COMPUTE VELOCITIES ######################################
	constexpr FLT cx[9] { 0, 1, 0,-1, 0, 1,-1,-1, 1};
	constexpr FLT cy[9] { 0, 0, 1, 0,-1, 1, 1,-1,-1};
	FLT ux{0.};
	FLT uy{0.};
	for (UINT i{0}; i<9; ++i){
		ux += f[i]*cx[i];
		uy += f[i]*cy[i];
	}
	ux *= rho_inv;
	uy *= rho_inv;

	// ###### COMPUTE EQUILIBIRUM DISTRIBUITON WRITE TO BUFFER ########
	constexpr FLT w[9] = {4./9.,1./9.,1./9.,1./9.,1./9.,1./36.,1./36.,1./36.,1./36.,};
	for (UINT i{0}; i<9; ++i){
		const FLT cx_ux{cx[i] * ux};
		const FLT cy_uy{cy[i] * uy};
		const FLT c_i_u_r = cx_ux + cy_uy;
		const FLT u_2 = ux * ux + uy * uy;
		const FLT f_eq{ w[i] * rho * (((FLT)1)+ ((FLT)3)*c_i_u_r+ ((FLT)4.5) * (c_i_u_r * c_i_u_r)- ((FLT)1.5) * u_2)};
		// compute neighbour to stream to
		const UINT x_n {
			(i==7 || i==6 || i==3) ? xl : (
			(i==4 || i==0 || i==2) ? x : xr)
		};
		const UINT y_n {
			(i==7 || i==4 || i==8) ? yd : (
			(i==3 || i==0 || i==1) ? y : yu)
		};
		VIEW(buf, x_n, y_n, i) = f[i] + OMEGA * (f_eq - f[i]);
	}
}


/// @brief Use a one-step, two-grid push-scheme to access memory in f, compute a new pdf value locally (collision) 
/// and then write write to neighbours (streaming) in memory-order. 
/// 
/// -> Wittmann, Zeiser, Hager, Wellein "Comparison of different propagation steps for lattice Boltzmann methods"
/// @param f field holding distribution PDF values
/// @param buf temporary write-only-buffer with dimensions equal to f that gets pointer-swapped at the end
void push_periodic_no_mpi(Dst_t& f, Dst_t& buf, DUI &nx, DUI &ny, DFL &om){
	Kokkos::parallel_for(
		"push periodic", 
		Kokkos::MDRangePolicy<Kokkos::Rank<2, Kokkos::Iterate::Left, Kokkos::Iterate::Left>>({0, 0}, {NX, NY}, {TX, TY}), 
		KOKKOS_LAMBDA(const UINT x, const UINT y){
			const UINT NX{nx()};
			const UINT NY{ny()};
			const FLT OMEGA{om()};
			// in the non-MPI version with no halo, conditionals are required for wrap-around
			const UINT xr{(x+1 == NX) ? (0)    : (x+1)};
			const UINT xl{(x   == 0 ) ? (NX-1) : (x-1)};
			const UINT yu{(y+1 == NY) ? (0)    : (y+1)};
			const UINT yd{(y   == 0 ) ? (NY-1) : (y-1)};
			#if UNROLL_LOOPS
				push_periodic(x,y,xl,xr,yd,yu,f,buf,OMEGA);
			#else 
				push_periodic_for(x,y,xl,xr,yd,yu,f,buf,OMEGA);
			#endif
		}
	);
}

void pull_periodic_no_mpi(Dst_t& f, Dst_t& buf, DUI &nx, DUI &ny, DFL &om){
	Kokkos::parallel_for(
		"pull periodic", 
		Kokkos::MDRangePolicy<Kokkos::Rank<2, Kokkos::Iterate::Left, Kokkos::Iterate::Left>>({0, 0}, {NX, NY}, {TX, TY}), 
		KOKKOS_LAMBDA(const UINT x, const UINT y){
			const UINT NX{nx()};
			const UINT NY{ny()};
			const FLT OMEGA{om()};
			// in the non-MPI version with no halo, conditionals are required for wrap-around
			const UINT xr{(x+1 == NX) ? (0)    : (x+1)};
			const UINT xl{(x   == 0 ) ? (NX-1) : (x-1)};
			const UINT yu{(y+1 == NY) ? (0)    : (y+1)};
			const UINT yd{(y   == 0 ) ? (NY-1) : (y-1)};
			#if UNROLL_LOOPS
				pull_periodic(x,y,xl,xr,yd,yu,f,buf,OMEGA);
			#else 
				pull_periodic_for(x,y,xl,xr,yd,yu,f,buf,OMEGA);
			#endif
		}
	);
}

void pull_periodic_inner(Dst_t& f, Dst_t& buf, DFL &om, UINT LX, UINT LY, UINT HX, UINT HY){
	Kokkos::parallel_for(
		"pull periodic", 
		Kokkos::MDRangePolicy<Kokkos::Rank<2, Kokkos::Iterate::Left, Kokkos::Iterate::Left>>({LX, LY}, {HX, HY}, {TX, TY}), 
		KOKKOS_LAMBDA(const UINT x, const UINT y){
			const FLT OMEGA{om()};
			// in this MPI-version there is a halo region 
			// so no explicit periodic boundary handling is required
			const UINT xr{x+1};
			const UINT xl{x-1};
			const UINT yu{y+1};
			const UINT yd{y-1};
			#if UNROLL_LOOPS
				pull_periodic(x, y, xl, xr, yd, yu, f, buf, OMEGA);
			#else 
				pull_periodic_for(x, y, xl, xr, yd, yu, f, buf, OMEGA);
			#endif
		}
	);
}

void push_periodic_inner(Dst_t& f, Dst_t& buf, DFL &om, UINT LX, UINT LY, UINT HX, UINT HY){
	Kokkos::parallel_for(
		"push periodic", 
		Kokkos::MDRangePolicy<Kokkos::Rank<2, Kokkos::Iterate::Left, Kokkos::Iterate::Left>>({LX, LY}, {HX, HY}, {TX, TY}), 
		KOKKOS_LAMBDA(const UINT x, const UINT y){
			const FLT OMEGA{om()};
			// in this MPI-version there is a halo region 
			// so no explicit periodic boundary handling is required
			const UINT xr{x+1};
			const UINT xl{x-1};
			const UINT yu{y+1};
			const UINT yd{y-1};
			#if UNROLL_LOOPS
				push_periodic(x, y, xl, xr, yd, yu, f, buf, OMEGA);
			#else 
				push_periodic_for(x, y, xl, xr, yd, yu, f, buf, OMEGA);
			#endif
		}
	);
}

void pull_periodic_outer(Dst_t& f, Dst_t& buf, DFL &om, DUI &nx, DUI &ny){
	// pull values on the ring defined by x=LX OR x=HX or y=LY or y=HY
	// without touching corners twice
	// -> 4x 1D launch of small kernels for each edge 
	// -> more launch overhead but perhaps better coalescing than fused kernels
	Kokkos::parallel_for(
		"pull top/bottom", 
		Kokkos::RangePolicy(1, NX-1), 
		KOKKOS_LAMBDA(const UINT x){
			const FLT OMEGA{om()}; 
			const UINT NY{ny()}; 
			const UINT xr{x+1}; const UINT xl{x-1};
			#if UNROLL_LOOPS
				// bottom
				pull_periodic(x, 1, xl, xr, 0, 2, f, buf, OMEGA);
				// top
				pull_periodic(x, NY-2, xl, xr, NY-3, NY-1, f, buf, OMEGA);
			#else 
				// bottom
				pull_periodic_for(x, 1, xl, xr, 0, 2, f, buf, OMEGA);
				// top
				pull_periodic_for(x, NY-2, xl, xr, NY-3, NY-1, f, buf, OMEGA);
			#endif
		}
	);
	Kokkos::parallel_for(
		"pull left/right", 
		// add an offset of one to avoid computing corners twice:
		Kokkos::RangePolicy(2, NY-2), // also note the absence of TX/TY
		KOKKOS_LAMBDA(const UINT y){
			const FLT OMEGA{om()}; 
			const UINT NX{nx()}; 
			const UINT yu{y+1}; const UINT yd{y-1};
			#if UNROLL_LOOPS
				// left
				pull_periodic(1, y, 0, 2, yd, yu, f, buf, OMEGA);
				// right
				pull_periodic(NX-2, y, NX-3, NX-1, yd, yu, f, buf, OMEGA);
			#else 
				// left
				pull_periodic_for(1, y, 0, 2, yd, yu, f, buf, OMEGA);
				// right
				pull_periodic_for(NX-2, y, NX-3, NX-1, yd, yu, f, buf, OMEGA);
			#endif
		}
	);
}

void push_periodic_outer(Dst_t& f, Dst_t& buf, DFL &om, DUI &nx, DUI &ny){
	// pull values on the ring defined by x=LX OR x=HX or y=LY or y=HY
	// without touching corners twice
	// -> 4x 1D launch of small kernels for each edge 
	// -> more launch overhead but perhaps better coalescing than fused kernels
	Kokkos::parallel_for(
		"push top/bottom", 
		Kokkos::RangePolicy(1, NX-1), 
		KOKKOS_LAMBDA(const UINT x){
			const FLT OMEGA{om()}; 
			const UINT NY{ny()}; 
			const UINT xr{x+1}; const UINT xl{x-1};
			#if UNROLL_LOOPS
				// bottom
				push_periodic(x, 1, xl, xr, 0, 2, f, buf, OMEGA);
				// top
				push_periodic(x, NY-2, xl, xr, NY-3, NY-1, f, buf, OMEGA);
			#else 
				// bottom
				push_periodic_for(x, 1, xl, xr, 0, 2, f, buf, OMEGA);
				// top
				push_periodic_for(x, NY-2, xl, xr, NY-3, NY-1, f, buf, OMEGA);
			#endif
		}
	);
	Kokkos::parallel_for(
		"push left/right", 
		// add an offset of one to avoid computing corners twice:
		Kokkos::RangePolicy(2, NY-2), // also note the absence of TX/TY
		KOKKOS_LAMBDA(const UINT y){
			const FLT OMEGA{om()}; 
			const UINT NX{nx()}; 
			const UINT yu{y+1}; const UINT yd{y-1};
			#if UNROLL_LOOPS
				// left
				push_periodic(1, y, 0, 2, yd, yu, f, buf, OMEGA);
				// right
				push_periodic(NX-2, y, NX-3, NX-1, yd, yu, f, buf, OMEGA);
			#else 
				// left
				push_periodic_for(1, y, 0, 2, yd, yu, f, buf, OMEGA);
				// right
				push_periodic_for(NX-2, y, NX-3, NX-1, yd, yu, f, buf, OMEGA);
			#endif
		}
	);
}

void push_lid_driven(Dst_t& f, Dst_t& buf, DUI &nx, DUI &ny, DFL &om, DFL &rho_eq, DFL &u_lid){
	Kokkos::parallel_for(
		"push interior", 
		Kokkos::MDRangePolicy<Kokkos::Rank<2, Kokkos::Iterate::Left, Kokkos::Iterate::Left>>({0, 0}, {NX, NY}, {TX, TY}),
		KOKKOS_LAMBDA(const UINT x, const UINT y){
			// load the distribution
			const FLT f_0 { VIEW(f, x, y, 0) };
			const FLT f_1 { VIEW(f, x, y, 1) };
			const FLT f_2 { VIEW(f, x, y, 2) };
			const FLT f_3 { VIEW(f, x, y, 3) };
			const FLT f_4 { VIEW(f, x, y, 4) };
			const FLT f_5 { VIEW(f, x, y, 5) };
			const FLT f_6 { VIEW(f, x, y, 6) };
			const FLT f_7 { VIEW(f, x, y, 7) };
			const FLT f_8 { VIEW(f, x, y, 8) };
			// load relevant global scalars
			const UINT NX{nx()};
			const UINT NY{ny()};
			const FLT OMEGA{om()};
			const FLT U_LID{u_lid()};
			const FLT RHO_EQ{rho_eq()};
			// calculate density and velocity
			const FLT rho { f_0 + f_1 + f_2 + f_3 + f_4 + f_5 + f_6 + f_7 + f_8 };
			constexpr FLT f1 {1.0};
			const FLT rho_inv {f1/rho};
		 	const FLT ux { (f_1 - f_3 + f_5 - f_6 - f_7 + f_8) * rho_inv };
			const FLT uy { (f_2 - f_4 + f_5 + f_6 - f_7 - f_8) * rho_inv };
			// calculate equilibirum values
			constexpr FLT f45 {4.5};
			constexpr FLT f15 {1.5};
			constexpr FLT f49 {4./9.};
			constexpr FLT f19 {1./9.};
			constexpr FLT f136 {1./36.};
			const FLT ux_2 {f45 * ux * ux};
			const FLT uy_2 {f45 * uy * uy};
			const FLT uymux {uy - ux};
			const FLT uymux_2 {f45 * uymux * uymux};
			const FLT uxpuy {ux + uy};
			const FLT uxpuy_2 {f45 * uxpuy * uxpuy};
			const FLT u_2_times_3_2 {f15 * (ux * ux + uy * uy)};
			const FLT w_4_9  {rho * f49};
			const FLT w_1_9  {rho * f19};
			const FLT w_1_36 {rho * f136};
			const FLT f_7_eq { f_7 + OMEGA * ((w_1_36 * (1- 3*uxpuy+ uxpuy_2- u_2_times_3_2)) - f_7) };
			const FLT f_3_eq { f_3 + OMEGA * ((w_1_9 * (1- 3*ux+ ux_2- u_2_times_3_2)) - f_3) };
			const FLT f_6_eq { f_6 + OMEGA * ((w_1_36 * (1+ 3*uymux+ uymux_2- u_2_times_3_2)) - f_6) };
			const FLT f_4_eq { f_4 + OMEGA * ((w_1_9 * (1- 3*uy+ uy_2- u_2_times_3_2)) - f_4) };
			const FLT f_0_eq { f_0 + OMEGA * ((w_4_9 * (1- u_2_times_3_2)) - f_0) };
			const FLT f_2_eq { f_2 + OMEGA * ((w_1_9 * (1+ 3*uy+ uy_2- u_2_times_3_2)) - f_2) };
			const FLT f_8_eq { f_8 + OMEGA * ((w_1_36 * (1- 3*uymux+ uymux_2- u_2_times_3_2)) - f_8) };
			const FLT f_1_eq { f_1 + OMEGA * ((w_1_9 * (1+ 3*ux+ ux_2- u_2_times_3_2)) - f_1) };
			const FLT f_5_eq { f_5 + OMEGA * ((w_1_36 * (1+ 3*uxpuy+ uxpuy_2- u_2_times_3_2)) - f_5) };

			// ###### STREAM VALUES, RESPECTING BOUNDARY CONDITIONS ###########
			// bounce back contributions from the wall to the corresponding channel of the current cell
			// (no overwrites should occur here so this is thread-safe)
			// 
			// 1 <-> 3     	| 6   2   5 |
			// 2 <-> 4     	|   \ | /   |
			// 5 <-> 7     	| 3 - 0 - 1 |
			// 6 <-> 8		|   / | \   |
			// 				| 7   4   8 |
			//
			// check if node is adjacent to any boundary
			const bool l{x==0};
			const bool r{x==(NX-1)};
			const bool t{y==(NY-1)};
			const bool b{y==0};
			const bool lb{l||b}; // bottom left corner
			const bool lt{l||t}; // top left corner
			const bool rb{r||b}; // bottom right corner
			const bool rt{r||t}; // top right corner
			// clamp x and y of neighbours to remain in bounds
			const UINT xl{(x>0)?(x-1):x};
			const UINT xr{(x<NX-1)?(x+1):x};
			const UINT yu{(y<NY-1)?(y+1):y};
			const UINT yd{(y>0)?(y-1):y};
			// calculate the change in distribution Δf due to the moving top wall:
			// 		Δf = -2 w_i ρ_w (c_i * u_w)/(c_s^2) 
			// 		1/c_s^2 = 1/(1/3) = 3
			// for i=6: uw*c_i = -1, total factor -2*(-1)/(1/3)=6
			// for i=8: uw*c_i =  1, total factor -2*( 1)/(1/3)=-6
			// since w_i = 1/36 either way we get +- 6 w_i ρ = +- 6/36 ρ = 1./6. ρ
			// const FLT df = {-2.0 * (1./36.) * RHO_EQ * (U_LID * (-1.))/(1./3.)};
			constexpr FLT f6 {6.};
			constexpr FLT f0 {0.};
			const FLT df = { t ? (RHO_EQ * U_LID / f6) : f0 };  
			// stream each channel to the corresponding neighbour node, or the current node if at a boundary
			VIEW(buf, lb?x:xl, lb?y:yd, lb?5:7) = f_7_eq;
			VIEW(buf, xl, y , l?1:3) 			= f_3_eq;
			VIEW(buf, lt?x:xl, lt?y:yu, lt?8:6) = f_6_eq + df; // add contribution from moving top wall
			VIEW(buf, x , yd, b?2:4) 			= f_4_eq;
			VIEW(buf, x , y , 0) 				= f_0_eq;
			VIEW(buf, x , yu, t?4:2) 			= f_2_eq;
			VIEW(buf, rb?x:xr, rb?y:yd, rb?6:8) = f_8_eq;
			VIEW(buf, xr, y , r?3:1) 			= f_1_eq;
			VIEW(buf, rt?x:xr, rt?y:yu, rt?7:5) = f_5_eq - df; // add contribution from moving top wall
		}
	);
}

bool run_simulation_single_node(){
	// create two buffers for use in two-grid, one-step updates
	#if COALESCING
		Dst_t f  ("f"  , Q, NY, NX); // only for reading
		Dst_t buf("buf", Q, NY, NX); // only for writing
	#else
		Dst_t f  ("f"  , NY, NX, Q); // only for reading
		Dst_t buf("buf", NY, NX, Q); // only for writing
	#endif
	// create buffer for velocity field in case it is needed for output
	Vel_t vel("vel", NX, NY);

	std::cerr<<"running on"<<NX<<"x"<<NY<<std::endl;
	// initialize device-side accessible scalar constants
	DUI nx = create_device_uint(NX);
	DUI ny = create_device_uint(NY);
	DFL om = create_device_float(OMEGA);
	DFL rho_init = create_device_float(RHO_INIT);
	DFL u = create_device_float(U_INIT);

	// establish initial conditions
	switch (SCENE){
		case SCENE_TYPE::LID_DRIVEN:
			init_rest(f, rho_init); break;
		default:
			// define additional device-side scalar constants: y-offset and global y size
			// this is simply 0 and NY for the single-node case
			DUI y_below = create_device_uint(0);		// global y offset: 0
			DUI y_glob = create_device_uint(NY);		// global y size: NY
			init_shearwave(f, rho_init, u, y_below, y_glob); break;
	}

	// main simulation loop
    for (UINT t{0}; t < STEPS; ++t){
		// write results to std::out
		if (OUT_EVERY_N > 0 && t % OUT_EVERY_N == 0){
			if (!output(vel, f, 0, 0, NX, NY, 0)){ return false; };
		}

		// do a single fused step of collision and streaming, tailored to the respective boundary conditions of the scene
		switch (SCENE){
			case SCENE_TYPE::LID_DRIVEN:
				push_lid_driven(f, buf, nx, ny, om, rho_init, u);
				break;
			default:
				// full domain, no halo:
				PUSH ? push_periodic_no_mpi(f, buf, nx, ny, om) : pull_periodic_no_mpi(f, buf, nx, ny, om);
				break;
		}

		// swap write buffer and read buffer
		auto temp = f;
		f = buf;
		buf = temp;
    };

	return true;

}

bool run_simulation_mpi(){
	// CREATE CARTESIAN COMMUNICATOR
	// https://wgropp.cs.illinois.edu/courses/cs598-s15/lectures/lecture28.pdf
	// define a cartesian grid communicator for efficient mapping of virtual to physical topology
	MPI_Comm grid;
	int size, rank;
	int dims[2] {0,0};  					// don't fix any dimension, create a grid
	int coords[2] {0,0};					// store coordinates of current rank
	int periods[2] {1,1};					// periodicity along each dimension
	int rank_L, rank_R, rank_D, rank_U, rank_DL, rank_DR, rank_UL, rank_UR;		// store ranks of each neighbouring node in 2D grid
    MPI_Comm_size(MPI_COMM_WORLD, &size); 	// get total number of processes
	MPI_Dims_create(size, 2, dims);			// get optimized number of processes along each dimension
	MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 1, &grid);
	MPI_Comm_rank(grid, &rank); 			// get own rank only now, w.r.t. grid, since reorder = 1 = true
	MPI_Cart_coords(grid, rank, 2, coords); // get the coordinates of the current rank
	// // get ranks of immediate neighbours
	// MPI_Cart_shift(grid, 0,  1, &rank_L, &rank_R);
	// MPI_Cart_shift(grid, 1,  1, &rank_D, &rank_U);
	// get ranks of neighbours, no need for % since periodic = 1 everywhere
	int nbr_coords[8][2] {
		{coords[0]-1, coords[1]  },
		{coords[0]+1, coords[1]  },
		{coords[0]  , coords[1]-1},
		{coords[0]  , coords[1]+1},
		{coords[0]-1, coords[1]-1},
		{coords[0]+1, coords[1]-1},
		{coords[0]-1, coords[1]+1},
		{coords[0]+1, coords[1]+1},
	};
	MPI_Cart_rank(grid, nbr_coords[0], &rank_L );
	MPI_Cart_rank(grid, nbr_coords[1], &rank_R );
	MPI_Cart_rank(grid, nbr_coords[2], &rank_D );
	MPI_Cart_rank(grid, nbr_coords[3], &rank_U );
	MPI_Cart_rank(grid, nbr_coords[4], &rank_DL);
	MPI_Cart_rank(grid, nbr_coords[5], &rank_DR);
	MPI_Cart_rank(grid, nbr_coords[6], &rank_UL);
	MPI_Cart_rank(grid, nbr_coords[7], &rank_UR);

	// determine the local and global size of the grid, as well as offsets from local to global coordinates
	const UINT nx_divisor {NX / dims[0]};
	const UINT nx_remainder {NX % dims[0]};
	const UINT ny_divisor {NY / dims[1]};
	const UINT ny_remainder {NY % dims[1]};
	// remember the global number of nodes along a direction
	const UINT NY_glob{NY};
	// - add one node to as many ranks as necessary to get rid of remainders
	// - also add +2 to account for the halo
	// - update the existing, global bindings for NX,NY with the rank-local values
	NX = nx_divisor + ((coords[0] < nx_remainder) ? 1 : 0) + 2;
	NY = ny_divisor + ((coords[1] < ny_remainder) ? 1 : 0) + 2;
	std::cerr<<nx_divisor<<" "<<nx_remainder<<" x "<<ny_divisor<<" "<<ny_remainder<<" coords: "<<coords[0]<<" "<<coords[1]<<std::endl;
	// determine the number of nodes in global coordinates that are below the lowest locally owned node along the y-axis
	const UINT Y_GLOBAL_BELOW{coords[1] * NY + ((coords[1] >= ny_remainder) ? ny_remainder : 0)};

	// create two buffers for use in two-grid, one-step updates
	#if COALESCING
		Dst_t f  ("f"  , Q, NY, NX); // only for reading
		Dst_t buf("buf", Q, NY, NX); // only for writing
	#else
		Dst_t f  ("f"  , NY, NX, Q); // only for reading
		Dst_t buf("buf", NY, NX, Q); // only for writing
	#endif

	// create buffer for velocity field in case it is needed for output
	Vel_t vel("vel", NX, NY);

	// initialize device-side accessible scalar constants
	DUI nx = create_device_uint(NX);
	DUI ny = create_device_uint(NY);
	DFL om = create_device_float(OMEGA);
	DFL rho_init = create_device_float(RHO_INIT);
	DFL u = create_device_float(U_INIT);

	// prepare MPI buffers, tags and request array
	Hlo_t top_buf("top halo buffer", (NX-2)*3);
	Hlo_t bot_buf("bot halo buffer", (NX-2)*3);
	Hlo_t lft_buf("left halo buffer", (NY-2)*3);
	Hlo_t rgt_buf("right halo buffer", (NY-2)*3);
	DFL lt("LT corner");
	DFL rt("RT corner");
	DFL lb("LB corner");
	DFL rb("RB corner");
	// use seperate buffers for sending and receiving on the host side to avoid race conditions and overwrites
	auto top_buf_send = Kokkos::create_mirror_view(Kokkos::DefaultHostExecutionSpace{}, top_buf);
	auto bot_buf_send = Kokkos::create_mirror_view(Kokkos::DefaultHostExecutionSpace{}, bot_buf);
	auto lft_buf_send = Kokkos::create_mirror_view(Kokkos::DefaultHostExecutionSpace{}, lft_buf);
	auto rgt_buf_send = Kokkos::create_mirror_view(Kokkos::DefaultHostExecutionSpace{}, rgt_buf);
	auto lt_send      = Kokkos::create_mirror_view(Kokkos::DefaultHostExecutionSpace{}, lt);
	auto rt_send      = Kokkos::create_mirror_view(Kokkos::DefaultHostExecutionSpace{}, rt);
	auto lb_send      = Kokkos::create_mirror_view(Kokkos::DefaultHostExecutionSpace{}, lb);
	auto rb_send      = Kokkos::create_mirror_view(Kokkos::DefaultHostExecutionSpace{}, rb);

	auto top_buf_recv = Kokkos::create_mirror_view(Kokkos::DefaultHostExecutionSpace{}, top_buf);
	auto bot_buf_recv = Kokkos::create_mirror_view(Kokkos::DefaultHostExecutionSpace{}, bot_buf);
	auto lft_buf_recv = Kokkos::create_mirror_view(Kokkos::DefaultHostExecutionSpace{}, lft_buf);
	auto rgt_buf_recv = Kokkos::create_mirror_view(Kokkos::DefaultHostExecutionSpace{}, rgt_buf);
	auto lt_recv      = Kokkos::create_mirror_view(Kokkos::DefaultHostExecutionSpace{}, lt);
	auto rt_recv      = Kokkos::create_mirror_view(Kokkos::DefaultHostExecutionSpace{}, rt);
	auto lb_recv      = Kokkos::create_mirror_view(Kokkos::DefaultHostExecutionSpace{}, lb);
	auto rb_recv      = Kokkos::create_mirror_view(Kokkos::DefaultHostExecutionSpace{}, rb);

	const int TAG_L_TO_R {0};
	const int TAG_R_TO_L {1};
	const int TAG_B_TO_T {2};
	const int TAG_T_TO_B {3};
	const int TAG_LT_TO_RB {4};
	const int TAG_RT_TO_LB {5};
	const int TAG_LB_TO_RT {6};
	const int TAG_RB_TO_LT {7};
	MPI_Request reqs[16];


	// initialize distribution field to given initial conditions
	// (buf need not be initialized since it is written to everywhere)
	switch (SCENE){
		case SCENE_TYPE::LID_DRIVEN:
			init_rest(f, rho_init); break;
		default:
			DUI y_below = create_device_uint(Y_GLOBAL_BELOW); // global y offset: how many nodes are below the rank-local y=0?
			DUI y_glob = create_device_uint(NY_glob);		  // global y size: how mnay nodes exist globally in y-direction?
			init_shearwave(f, rho_init, u, y_below, y_glob); break;
	}

	// main simulation loop
    for (UINT t{0}; t < STEPS; ++t){
		// COMMUNICATE:
		// TODO: use not own but corresponding neighbours' rank
		// receive and send halo slices asynchronously, collecting the request receipts
		// https://www.osti.gov/servlets/purl/1818024

		// STEP 1: POST RECEIVES
		MPI_Irecv(static_cast<void*>(top_buf_recv.data()), top_buf_recv.size(), MFLOAT, rank_U , TAG_B_TO_T  , grid, &reqs[ 0]);
		MPI_Irecv(static_cast<void*>(bot_buf_recv.data()), bot_buf_recv.size(), MFLOAT, rank_D , TAG_T_TO_B  , grid, &reqs[ 1]);
		MPI_Irecv(static_cast<void*>(lft_buf_recv.data()), lft_buf_recv.size(), MFLOAT, rank_L , TAG_R_TO_L  , grid, &reqs[ 2]);
		MPI_Irecv(static_cast<void*>(rgt_buf_recv.data()), rgt_buf_recv.size(), MFLOAT, rank_R , TAG_L_TO_R  , grid, &reqs[ 3]);
		MPI_Irecv(static_cast<void*>(lt_recv.data())     , lt_recv.size()     , MFLOAT, rank_UL, TAG_RB_TO_LT, grid, &reqs[ 4]);
		MPI_Irecv(static_cast<void*>(rt_recv.data())     , rt_recv.size()     , MFLOAT, rank_UR, TAG_LB_TO_RT, grid, &reqs[ 5]);
		MPI_Irecv(static_cast<void*>(lb_recv.data())     , lb_recv.size()     , MFLOAT, rank_DL, TAG_RT_TO_LB, grid, &reqs[ 6]);
		MPI_Irecv(static_cast<void*>(rb_recv.data())     , rb_recv.size()     , MFLOAT, rank_DR, TAG_LT_TO_RB, grid, &reqs[ 7]);

		// STEP 2: PACKING
		// 		| 6   2   5 |
		// 		|   \ | /   |
		// 		| 3 - 0 - 1 |
		// 		|   / | \   |
		// 		| 7   4   8 |
		Kokkos::parallel_for("top/bot packing", NX-2, KOKKOS_LAMBDA(const UINT x){
			// top: y=NY-2 -> bot: y=0
			const UINT NY{ny()};
			const UINT OFF{nx()-2};
			top_buf(x      ) = VIEW(f, x+1, NY-2, 6);
			top_buf(x+  OFF) = VIEW(f, x+1, NY-2, 2);
			top_buf(x+2*OFF) = VIEW(f, x+1, NY-2, 5);
			// bot: y=1 -> top: y=NY-1
			bot_buf(x      ) = VIEW(f, x+1, 1, 7);
			bot_buf(x+  OFF) = VIEW(f, x+1, 1, 4);
			bot_buf(x+2*OFF) = VIEW(f, x+1, 1, 8);
		});
		Kokkos::parallel_for("lft/rgt packing", NY-2, KOKKOS_LAMBDA(const UINT y){
			const UINT NX{nx()};
			const UINT OFF{ny()-2};
			// left: x=1 ->  right: x=NX-1
			lft_buf(y      ) = VIEW(f, 1, y+1, 7);
			lft_buf(y+  OFF) = VIEW(f, 1, y+1, 3);
			lft_buf(y+2*OFF) = VIEW(f, 1, y+1, 6);
			// right: x=NX-2 ->  left: x=0
			rgt_buf(y      ) = VIEW(f, NX-2, y+1, 8);
			rgt_buf(y+  OFF) = VIEW(f, NX-2, y+1, 1);
			rgt_buf(y+2*OFF) = VIEW(f, NX-2, y+1, 5);
		});
		Kokkos::parallel_for("corner exchange", 1, KOKKOS_LAMBDA(const UINT i){
			const UINT NX{nx()};
			const UINT NY{ny()};
			// lt -> rb
			lt() = VIEW(f, 1, NY-2, 6);
			// rt -> lb
			rt() = VIEW(f, NX-2, NY-2, 5);
			// lb -> rt
			lb() = VIEW(f, 1, 1, 7);
			// rb -> lt
			rb() = VIEW(f, NX-2, 1, 8);
		});
		Kokkos::fence();
		Kokkos::deep_copy(top_buf_send, top_buf);
		Kokkos::deep_copy(bot_buf_send, bot_buf);
		Kokkos::deep_copy(lft_buf_send, lft_buf);
		Kokkos::deep_copy(rgt_buf_send, rgt_buf);
		Kokkos::deep_copy(lt_send     , lt     );
		Kokkos::deep_copy(rt_send     , rt     );
		Kokkos::deep_copy(lb_send     , lb     );
		Kokkos::deep_copy(rb_send     , rb     );
		Kokkos::fence();


		// 3. POST SENDS
		MPI_Isend(static_cast<void*>(top_buf_send.data()), top_buf_send.size(), MFLOAT, rank_U , TAG_T_TO_B  , grid, &reqs[ 8]);
		MPI_Isend(static_cast<void*>(bot_buf_send.data()), bot_buf_send.size(), MFLOAT, rank_D , TAG_B_TO_T  , grid, &reqs[ 9]);
		MPI_Isend(static_cast<void*>(lft_buf_send.data()), lft_buf_send.size(), MFLOAT, rank_L , TAG_L_TO_R  , grid, &reqs[10]);
		MPI_Isend(static_cast<void*>(rgt_buf_send.data()), rgt_buf_send.size(), MFLOAT, rank_R , TAG_R_TO_L  , grid, &reqs[11]);
		MPI_Isend(static_cast<void*>(lt_send.data())     , lt_send.size()     , MFLOAT, rank_UL, TAG_LT_TO_RB, grid, &reqs[12]);
		MPI_Isend(static_cast<void*>(rt_send.data())     , rt_send.size()     , MFLOAT, rank_UR, TAG_RT_TO_LB, grid, &reqs[13]);
		MPI_Isend(static_cast<void*>(lb_send.data())     , lb_send.size()     , MFLOAT, rank_DL, TAG_LB_TO_RT, grid, &reqs[14]);
		MPI_Isend(static_cast<void*>(rb_send.data())     , rb_send.size()     , MFLOAT, rank_DR, TAG_RB_TO_LT, grid, &reqs[15]);

		// // SENDRECV VARIANT (does not work at the moment)
		// MPI_Status ignore;
		// MPI_Sendrecv(static_cast<void*>(top_buf_send.data()), top_buf_send.size(), MFLOAT, rank_U , 0  , 
		// 			static_cast<void*>(bot_buf_recv.data()), bot_buf_recv.size(), MFLOAT, rank_D , 0  ,grid, &ignore);
		// MPI_Sendrecv(static_cast<void*>(bot_buf_send.data()), bot_buf_send.size(), MFLOAT, rank_D , 0  , 
		// 			static_cast<void*>(top_buf_recv.data()), top_buf_recv.size(), MFLOAT, rank_U , 0  ,grid, &ignore);
		// MPI_Sendrecv(static_cast<void*>(lft_buf_send.data()), lft_buf_send.size(), MFLOAT, rank_L , 0  ,
		// 			static_cast<void*>(rgt_buf_recv.data()), rgt_buf_recv.size(), MFLOAT, rank_R , 0  ,grid, &ignore);
		// MPI_Sendrecv(static_cast<void*>(rgt_buf_send.data()), rgt_buf_send.size(), MFLOAT, rank_R , 0  ,
		// 			static_cast<void*>(lft_buf_recv.data()), lft_buf_recv.size(), MFLOAT, rank_L , 0  ,grid, &ignore);
		// MPI_Sendrecv(static_cast<void*>(lt_send.data())     , lt_send.size()     , MFLOAT, rank_UL, 0,
		// 			static_cast<void*>(rb_recv.data())     , rb_recv.size()     , MFLOAT, rank_DR, 0,grid, &ignore);
		// MPI_Sendrecv(static_cast<void*>(rt_send.data())     , rt_send.size()     , MFLOAT, rank_UR, 0,
		// 			static_cast<void*>(lb_recv.data())     , lb_recv.size()     , MFLOAT, rank_DL, 0,grid, &ignore);
		// MPI_Sendrecv(static_cast<void*>(lb_send.data())     , lb_send.size()     , MFLOAT, rank_DL, 0,
		// 			static_cast<void*>(rt_recv.data())     , rt_recv.size()     , MFLOAT, rank_UR, 0,grid, &ignore);
		// MPI_Sendrecv(static_cast<void*>(rb_send.data())     , rb_send.size()     , MFLOAT, rank_DR, 0,
		// 			static_cast<void*>(lt_recv.data())     , lt_recv.size()     , MFLOAT, rank_UL, 0,grid, &ignore);

		// WORK
		switch (SCENE){
			case SCENE_TYPE::LID_DRIVEN:
				std::cerr << "Lid driven cavity currently not supported when using MPI" << std::endl;
				return false;
			default:
				PUSH ? push_periodic_inner(f, buf, om, 2, 2, NX-2, NY-2) : pull_periodic_inner(f, buf, om, 2, 2, NX-2, NY-2);
				break;
		}

		// 4. WAIT FOR COMMUNICATION TO FINISH
		MPI_Waitall(16, reqs, MPI_STATUSES_IGNORE);

		// 5. UNPACK
		Kokkos::fence();
		Kokkos::deep_copy(top_buf, top_buf_recv);
		Kokkos::deep_copy(bot_buf, bot_buf_recv);
		Kokkos::deep_copy(lft_buf, lft_buf_recv);
		Kokkos::deep_copy(rgt_buf, rgt_buf_recv);
		Kokkos::deep_copy(lt     , lt_recv     );
		Kokkos::deep_copy(rt     , rt_recv     );
		Kokkos::deep_copy(lb     , lb_recv     );
		Kokkos::deep_copy(rb     , rb_recv     );
		Kokkos::fence();
		Kokkos::parallel_for("top/bot unpacking", NX-2, KOKKOS_LAMBDA(const UINT x){
			// top: y=NY-2 -> bot: y=0
			const UINT NY{ny()};
			const UINT OFF{nx()-2};
			VIEW(f, x+1, 0, 6) = bot_buf(x      );
			VIEW(f, x+1, 0, 2) = bot_buf(x+  OFF);
			VIEW(f, x+1, 0, 5) = bot_buf(x+2*OFF);
			// bot: y=1 -> top: y=NY-1
			VIEW(f, x+1, NY-1, 7) = top_buf(x      );
			VIEW(f, x+1, NY-1, 4) = top_buf(x+  OFF);
			VIEW(f, x+1, NY-1, 8) = top_buf(x+2*OFF);
		});
		Kokkos::parallel_for("lft/rgt unpacking", NY-2, KOKKOS_LAMBDA(const UINT y){
			const UINT NX{nx()};
			const UINT OFF{ny()-2};
			// left: x=1 ->  right: x=NX-1
			VIEW(f, NX-1, y+1, 7) = rgt_buf(y      );
			VIEW(f, NX-1, y+1, 3) = rgt_buf(y+  OFF);
			VIEW(f, NX-1, y+1, 6) = rgt_buf(y+2*OFF);
			// right: x=NX-2 ->  left: x=0
			VIEW(f, 0, y+1, 8) = lft_buf(y      );
			VIEW(f, 0, y+1, 1) = lft_buf(y+  OFF);
			VIEW(f, 0, y+1, 5) = lft_buf(y+2*OFF);
		});
		Kokkos::parallel_for("corner exchange", 1, KOKKOS_LAMBDA(const UINT i){
			const UINT NX{nx()};
			const UINT NY{ny()};
			// lt -> rb
			VIEW(f, NX-1, 0, 6) = rb();
			// rt -> lb
			VIEW(f, 0, 0, 5) = lb();
			// lb -> rt
			VIEW(f, NX-1, NY-1, 7) = rt();
			// rb -> lt
			VIEW(f, 0, NY-1, 8) = lt();
		});

		// write results to std::out
		if (OUT_EVERY_N > 0 && t % OUT_EVERY_N == 0){
			if (!output(vel, f, 1, 1, NX-1, NY-1, rank)){ return false; };
		}

		// do a single fused step of collision and streaming, tailored to the respective boundary conditions of the scene
		switch (SCENE){
			case SCENE_TYPE::LID_DRIVEN:
				std::cerr << "Lid driven cavity currently not supported when using MPI" << std::endl;
				return false;
			default:
				// push_periodic(f, buf, nx, ny, om); break;
				PUSH ? push_periodic_outer(f, buf, om, nx, ny) : pull_periodic_outer(f, buf, om, nx, ny); 
				// full domain, no halo:
				// pull_periodic_inner(f, buf, om, 1, 1, NX-1, NY-1);
				break;
		}

		// swap write buffer and read buffer
		auto temp = f;
		f = buf;
		buf = temp;
    };

	return true;
}


// MAIN ENTRY POINT

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);
	// parse command line arguments to adjust NX, NY, OMEGA, ...
	parse_args(argc, argv);
	// Initialize Kokkos 
    Kokkos::initialize(argc, argv);
	// Kokkos::print_configuration(std::cout);

	// new scope to make sure deallocation precedes finalize()
	bool success {true};
	{
		// Run the simulation
		auto start {std::chrono::high_resolution_clock::now()};
		success = USE_MPI ? run_simulation_mpi() : run_simulation_single_node();
		auto end {std::chrono::high_resolution_clock::now()};
		// print the MLUPS count
		auto span {std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count()};
		auto mlups = (NX*NY*STEPS*1000) / span;
		std::cerr << std::endl 
			<< NX*NY*STEPS << " lattice updates in |" 
			<< span << "|ns => " 
			<< mlups << " MLUPS"
			<< std::endl;
	}
	(*OUT_STREAM) << std::flush;
    Kokkos::finalize();
    MPI_Finalize();
    return success ? 0 : 1;
}
