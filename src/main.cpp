
#include <iostream>
#include <stdlib.h>

#include <argparse/argparse.hpp>
#include "Kokkos_Core.hpp"

// TYPES

#define FLOAT float
#define SIN sinf
#define SQRT sqrtf
#define INT int64_t
#define UINT uint64_t
#define VIEW(view, i, j, k) view(i, j, k)

// PARAMETERS

enum OUTPUT_TYPE{
	/// @brief no output
	NONE,
	/// @brief the maximum velocity magnitude across all nodes
	MAX_VEL,
	/// @brief the entire velocity field
	VEL_FIELD,
};
/// @brief the type of output printed to std::out when dumping data
OUTPUT_TYPE OUTPUT {OUTPUT_TYPE::NONE};

/// @brief dump simulation results to std::out every so many timesteps
UINT OUT_EVERY_N{100};
/// @brief number of total time steps of the simulation
UINT STEPS{50000};
/// @brief # grid points in y-direction
UINT NY;
/// @brief # grid points in x-direction
UINT NX;
/// @brief number of discrete velocity directions
const UINT Q{9};

/// @brief interpolation coefficient towards local equilibrium distribution
FLOAT OMEGA;
/// @brief initial density
const FLOAT RHO_INIT{0.5};
/// @brief initial distribution values
const FLOAT U_INIT{0.2};

// FIELD TYPES

/// Field of density values (x,y)
using Den_t = Kokkos::View<FLOAT**, Kokkos::LayoutRight>;

/// Field of velocity values (dir,x,y)
using Vel_t = Kokkos::View<FLOAT**[2], Kokkos::LayoutRight>;

/// Host-side field of velocity values (dir,x,y)
using Vel_t_host = Kokkos::View<FLOAT**[2], Kokkos::LayoutRight, Kokkos::HostSpace::device_type, Kokkos::Experimental::DefaultViewHooks>;

/// Field of distribution values (dir,x,y)
using Dst_t = Kokkos::View<FLOAT**[Q], Kokkos::LayoutRight>;

/// Field of coordinates of bounce-back, no-slip walls
using Bdy_t = Kokkos::View<UINT*[2], Kokkos::LayoutRight>;


/// @brief A scalar, unsigned int
using SUI = Kokkos::View<UINT>;

/// @brief A scalar float
using SFL = Kokkos::View<FLOAT>;


// FUNCTIONS

/// @brief Use a one-step, two-grid push-scheme to access coherent memory in f, compute a new pdf value locally (collision) 
/// and then write write to neighbours (streaming) in memory-order. 
/// 
/// -> Wittmann, Zeiser, Hager, Wellein "Comparison of different propagation steps for lattice Boltzmann methods"
/// @param f field holding distribution PDF values
/// @param buf temporary write-only-buffer with dimensions equal to f that gets pointer-swapped at the end
void push_periodic(Dst_t& f, Dst_t& buf, SUI &nx, SUI &ny, SFL &om){
	Kokkos::parallel_for(
		"push periodic", 
		Kokkos::MDRangePolicy({0, 0}, {NX, NY}), 
		KOKKOS_LAMBDA(const UINT x, const UINT y){

			// ###### READ VALUES #############################################
			// load f_i values into registers to avoid coalescing requirements for multiple reads
			// - we prefer coherent reads over coherent writes since they can be delayed and buffered to hide latency
			//   while buffered reads stall computation, so this section reads immediately subsequent memory
			const FLOAT f_0 { VIEW(f, x, y, 0) };
			const FLOAT f_1 { VIEW(f, x, y, 1) };
			const FLOAT f_2 { VIEW(f, x, y, 2) };
			const FLOAT f_3 { VIEW(f, x, y, 3) };
			const FLOAT f_4 { VIEW(f, x, y, 4) };
			const FLOAT f_5 { VIEW(f, x, y, 5) };
			const FLOAT f_6 { VIEW(f, x, y, 6) };
			const FLOAT f_7 { VIEW(f, x, y, 7) };
			const FLOAT f_8 { VIEW(f, x, y, 8) };

			// ###### COMPUTE DENSITIES #######################################
			// collect density contributions from all directions
			const FLOAT rho = f_0 + f_1 + f_2 + f_3 + f_4 + f_5 + f_6 + f_7 + f_8;
			const FLOAT rho_inv = 1./rho;

			// ###### COMPUTE VELOCITIES ######################################
			// contributions to x-velocities come from channels 1, 3, 5, 6, 7, 8
			/// 3,6,7 get a minus sign, 0,2,4 don't contribute
		 	const FLOAT ux = (f_1 - f_3 + f_5 - f_6 - f_7 + f_8) * rho_inv;
			// contributions to y-velocities come from channels 2, 4, 5, 6, 7, 8
			/// 4,7,8 get a minus sign, 0,1,3 don't contribute
			const FLOAT uy = (f_2 - f_4 + f_5 + f_6 - f_7 - f_8) * rho_inv;

			// ###### COMPUTE EQUILIBIRUM DISTRIBUITON AND STREAM #############
			// What the compiler ought to already do:
			// - factor out common subexpressions
			// - avoid pointless arithmetic like *1, *0, use - instead of *(-1)
			// - use conditionals instead of %
			// Other optimizations:
			// - reorder writes so they access subsequent memory

			const UINT NX{nx()};
			const UINT NY{ny()};
			const UINT xr{(x+1 == NX) ? (0)    : (x+1)};
			const UINT xl{(x   == 0 ) ? (NX-1) : (x-1)};
			const UINT yu{(y+1 == NY) ? (0)    : (y+1)};
			const UINT yd{(y   == 0 ) ? (NY-1) : (y-1)};


			// common subexpressions
			const FLOAT ux_2 = 4.5 * ux * ux;
			const FLOAT uy_2 = 4.5 * uy * uy;
				// cross terms
			const FLOAT uymux = uy - ux;
			const FLOAT uymux_2 = 4.5 * uymux * uymux;
			const FLOAT uxpuy = ux + uy;
			const FLOAT uxpuy_2 = 4.5 * uxpuy * uxpuy;
			const FLOAT u_2_times_3_2 = 1.5 * (ux * ux + uy * uy);
			// weights and density
			const FLOAT w_4_9  = rho * 4./9.;
			const FLOAT w_1_9  = rho * 1./9.;
			const FLOAT w_1_36 = rho * 1./36.;

			// to order writes to be as coherent as possible, write to xl, x xr and within those to yd, y yu:
			// 7 3 6 -> 4 0 2 -> 8 1 5 (bottom to top, left to right)
			const FLOAT OMEGA{om()};
			VIEW(buf, xl, yd, 7) = f_7 + OMEGA * ((w_1_36 * (1- 3*uxpuy+ uxpuy_2- u_2_times_3_2)) - f_7);
			VIEW(buf, xl, y , 3) = f_3 + OMEGA * ((w_1_9 * (1- 3*ux+ ux_2- u_2_times_3_2)) - f_3);
			VIEW(buf, xl, yu, 6) = f_6 + OMEGA * ((w_1_36 * (1+ 3*uymux+ uymux_2- u_2_times_3_2)) - f_6);
			VIEW(buf, x , yd, 4) = f_4 + OMEGA * ((w_1_9 * (1- 3*uy+ uy_2- u_2_times_3_2)) - f_4);
			VIEW(buf, x , y , 0) = f_0 + OMEGA * ((w_4_9 * (1- u_2_times_3_2)) - f_0);
			VIEW(buf, x , yu, 2) = f_2 + OMEGA * ((w_1_9 * (1+ 3*uy+ uy_2- u_2_times_3_2)) - f_2);
			VIEW(buf, xr, yd, 8) = f_8 + OMEGA * ((w_1_36 * (1- 3*uymux+ uymux_2- u_2_times_3_2)) - f_8);
			VIEW(buf, xr, y , 1) = f_1 + OMEGA * ((w_1_9 * (1+ 3*ux+ ux_2- u_2_times_3_2)) - f_1);
			VIEW(buf, xr, yu, 5) = f_5 + OMEGA * ((w_1_36 * (1+ 3*uxpuy+ uxpuy_2- u_2_times_3_2)) - f_5);
		}
	);
	auto temp = f;
    f = buf;
    buf = temp;
}


void push_solid_boundary(Dst_t& f, Dst_t& buf, SUI &nx, SUI &ny, SFL &om){
	Kokkos::parallel_for(
		"push solid_boundary", 
		Kokkos::MDRangePolicy({0, 0}, {NX, NY}), 
		KOKKOS_LAMBDA(const UINT x, const UINT y){

			// ###### READ VALUES #############################################
			// load f_i values into registers to avoid coalescing requirements for multiple reads
			const FLOAT f_0 { VIEW(f, x, y, 0) };
			const FLOAT f_1 { VIEW(f, x, y, 1) };
			const FLOAT f_2 { VIEW(f, x, y, 2) };
			const FLOAT f_3 { VIEW(f, x, y, 3) };
			const FLOAT f_4 { VIEW(f, x, y, 4) };
			const FLOAT f_5 { VIEW(f, x, y, 5) };
			const FLOAT f_6 { VIEW(f, x, y, 6) };
			const FLOAT f_7 { VIEW(f, x, y, 7) };
			const FLOAT f_8 { VIEW(f, x, y, 8) };

			// ###### COMPUTE DENSITIES #######################################
			// collect density contributions from all directions
			const FLOAT rho {f_0 + f_1 + f_2 + f_3 + f_4 + f_5 + f_6 + f_7 + f_8};
			const FLOAT rho_inv {1./rho};

			// ###### COMPUTE VELOCITIES ######################################
			// contributions to x-velocities come from channels 1, 3, 5, 6, 7, 8
			/// 3,6,7 get a minus sign, 0,2,4 don't contribute
		 	const FLOAT ux {(f_1 - f_3 + f_5 - f_6 - f_7 + f_8) * rho_inv};
			// contributions to y-velocities come from channels 2, 4, 5, 6, 7, 8
			/// 4,7,8 get a minus sign, 0,1,3 don't contribute
			const FLOAT uy {(f_2 - f_4 + f_5 + f_6 - f_7 - f_8) * rho_inv};

			// ###### COMPUTE EQUILIBIRUM DISTRIBUITON ########################
			// common subexpressions
			const FLOAT ux_2 {4.5 * ux * ux};
			const FLOAT uy_2 {4.5 * uy * uy};
				// cross terms
			const FLOAT uymux {uy - ux};
			const FLOAT uymux_2 {4.5 * uymux * uymux};
			const FLOAT uxpuy {ux + uy};
			const FLOAT uxpuy_2 {4.5 * uxpuy * uxpuy};
			const FLOAT u_2_times_3_2 {1.5 * (ux * ux + uy * uy)};
			// weights and density
			const FLOAT w_4_9  {rho * 4./9.};
			const FLOAT w_1_9  {rho * 1./9.};
			const FLOAT w_1_36 {rho * 1./36.};

			// equilibrium distribution values
			const FLOAT OMEGA{om()};
			const FLOAT f_eq_0 {f_0 + OMEGA * ((w_4_9 * ( 1 - u_2_times_3_2 )) - f_0)};
			const FLOAT f_eq_1 {f_1 + OMEGA * ((w_1_9 * (1+ 3*ux+ ux_2- u_2_times_3_2 )) - f_1)};
			const FLOAT f_eq_2 {f_2 + OMEGA * ((w_1_9 * (1+ 3*uy+ uy_2- u_2_times_3_2 )) - f_2)};
			const FLOAT f_eq_3 {f_3 + OMEGA * ((w_1_9 * (1- 3*ux + ux_2 - u_2_times_3_2 )) - f_3)};
			const FLOAT f_eq_4 {f_4 + OMEGA * ((w_1_9 * (1- 3*uy+ uy_2- u_2_times_3_2 )) - f_4)};
			const FLOAT f_eq_5 {f_5 + OMEGA * ((w_1_36 * (1+ 3*uxpuy+ uxpuy_2- u_2_times_3_2)) - f_5)};
			const FLOAT f_eq_6 {f_6 + OMEGA * ((w_1_36 * (1+ 3*uymux+ uymux_2- u_2_times_3_2 )) - f_6)};
			const FLOAT f_eq_7 {f_7 + OMEGA * ((w_1_36 * (1- 3*uxpuy+ uxpuy_2- u_2_times_3_2 )) - f_7)};
			const FLOAT f_eq_8 {f_8 + OMEGA * ((w_1_36 * (1- 3*uymux+ uymux_2- u_2_times_3_2 )) - f_8)};

			// ###### STREAM VALUES, RESPECTING BOUNDARY CONDITIONS ###########
			// bounce back contributions from the wall to the corresponding neighbouring cell
			// (no overwrites should occur here so this is inherently thread-safe)
			// 
			// 1 <-> 3     	| 6   2   5 |
			// 2 <-> 4     	|   \ | /   |
			// 5 <-> 7     	| 3 - 0 - 1 |
			// 6 <-> 8		|   / | \   |
			// 				| 7   4   8 |
			//
			// Compute target channels for each channel depending on boundary
			const UINT NX{nx()};
			const UINT NY{ny()};

			// most cells will not be on the boundary and can enjoy the fast path:
			if (x>0 && x<NX-1 && y>0 && y<NY-1) {
				// interior nodes:
				const UINT xr{(x+1 == NX) ? (0)    : (x+1)};
				const UINT xl{(x   == 0 ) ? (NX-1) : (x-1)};
				const UINT yu{(y+1 == NY) ? (0)    : (y+1)};
				const UINT yd{(y   == 0 ) ? (NY-1) : (y-1)};
				VIEW(buf, xl, yd, 7) = f_eq_7;
				VIEW(buf, xl, y , 3) = f_eq_3;
				VIEW(buf, xl, yu, 6) = f_eq_6;
				VIEW(buf, x , yd, 4) = f_eq_4;
				VIEW(buf, x , y , 0) = f_eq_0;
				VIEW(buf, x , yu, 2) = f_eq_2;
				VIEW(buf, xr, yd, 8) = f_eq_8;
				VIEW(buf, xr, y , 1) = f_eq_1;
				VIEW(buf, xr, yu, 5) = f_eq_5;
			} else {
				// only for nodes on the boundary, additional conditionals are required:
				const UINT c1{ x == NX-1 ? 3 : 1 };
				const UINT c5{ x == NX-1 || y==NY-1 ? 7 : 5 };
				const UINT c8{ x == NX-1 || y==0 ? 6 : 8 };

				const UINT c2{ y == NY-1 ? 4 : 2 };
				const UINT c0{0};
				const UINT c4{ y == 0 ? 2 : 4 };

				const UINT c6{ x == 0 || y==NY-1 ? 8 : 6 };
				const UINT c3{ x == 0 ? 1 : 3 };
				const UINT c7{ x == 0 || y==0 ? 5 : 7 };

				// stream
				VIEW(buf, x==NX-1?x:x+1, y, c1) = f_eq_1;
				VIEW(buf, x==NX-1?x:x+1, y==NY-1?y:y+1, c5) = f_eq_5;
				VIEW(buf, x==NX-1?x:x+1, y==0?y:y-1, c8) = f_eq_8;

				VIEW(buf, x, y==NY-1?y:y+1, c2) = f_eq_2;
				VIEW(buf, x, y, c0) = f_eq_0;
				VIEW(buf, x, y==0?y:y-1, c4) = f_eq_4;

				VIEW(buf, x==0?x:x-1, y==NY-1?y:y+1, c6) = f_eq_6;
				VIEW(buf, x==0?x:x-1, y, c3) = f_eq_3;
				VIEW(buf, x==0?x:x-1, y==0?y:y-1, c7) = f_eq_7;
			}
		}
	);
	auto temp = f;
    f = buf;
    buf = temp;
}

// void stream(Dst_t& f, Dst_t& buf){}

/// @brief Perform a streaming step on the lattice, moving each f_i in it's corresponding direction
/// @param f the distribution to stream
/// @param buf a buffer used to avoid overwriting values used by other threads
void stream(Dst_t& f, Dst_t& buf, SUI &nx, SUI &ny){
    Kokkos::parallel_for(
		"streaming step", 
		Kokkos::MDRangePolicy({0, 0, 0}, {NX, NY, Q}), 
		KOKKOS_LAMBDA(const UINT x, const UINT y, const UINT dir){
			/// @brief enumerates the discrete velocity vector in the 
			/// i-th discrete direction using a pair of integers
			const INT dx[9] = {0, 1, 0,-1, 0, 1,-1,-1, 1,};
			const INT dy[9] = {0, 0, 1, 0,-1, 1, 1,-1,-1,};
			// compute position of the cell to take f from
			const INT new_x = x + dx[dir];
			const INT new_y = y + dy[dir];
			// if the offset position is out of bounds, wrap around
			// -> use conditionals instead of modulo here since % is slow (?)
			// - this only works as intended if |dx| < NX, |dy| < NY
			//   which should be the case since in this setting |dx|,|dy| <= 1
			const UINT NX{nx()};
			const UINT NY{ny()};
			const INT wx = (new_x < 0) ? (new_x + NX) : ((new_x >= NX) ? (new_x - NX) : new_x);
			const INT wy = (new_y < 0) ? (new_y + NY) : ((new_y >= NY) ? (new_y - NY) : new_y);
			// write the moved f to a buffer 
			// -> in-place modification is unsafe without synchronization
			VIEW(buf, x, y, dir) = VIEW(f, wx, wy, dir);
		}
	);
	// perform pointer swap to move the output buffer
    auto temp = f;
    f = buf;
    buf = temp;
}


void compute_densities(Den_t &rho, Dst_t &f){
	Kokkos::parallel_for(
		"compute densities", 
		Kokkos::MDRangePolicy({0, 0}, {NX, NY}), 
		KOKKOS_LAMBDA(const UINT x, const UINT y){
			// collect density contributions from all directions
			FLOAT sum = 0.;
			for (UINT dir{0}; dir<Q; ++dir){
				sum += VIEW(f, x, y, dir);
			};
			// store the result at the corresponding discretization point
			rho(x,y) = sum;
		}
	);
}

void compute_velocities(Vel_t &vel, Den_t &rho, Dst_t &f){
	Kokkos::parallel_for(
		"compute velocities", 
		Kokkos::MDRangePolicy({0, 0}, {NX, NY}), 
		KOKKOS_LAMBDA(const UINT x, const UINT y){
			// load f_i values into registers to avoid coalescing requirements for multiple accesses
			// -> note that f_0 is not loaded since it would be weighted with a zero velocity anyways
			FLOAT f_1 { VIEW(f, x, y, 1) };
			FLOAT f_2 { VIEW(f, x, y, 2) };
			FLOAT f_3 { VIEW(f, x, y, 3) };
			FLOAT f_4 { VIEW(f, x, y, 4) };
			FLOAT f_5 { VIEW(f, x, y, 5) };
			FLOAT f_6 { VIEW(f, x, y, 6) };
			FLOAT f_7 { VIEW(f, x, y, 7) };
			FLOAT f_8 { VIEW(f, x, y, 8) };

			// OPTIMIZATION:
			// - From here on, unroll the loop over discrete directions, since only 6/9 entries of c 
			//   contribute to each component of the velocity, the others are zero.
			// - Also, negation can then be used instead of generic multiplication with -1.0

			// get contributions to x-velocities
			// these come from channels 1, 3, 5, 6, 7, 8
			/// 3,6,7 get a minus sign, 0,2,4 don't contribute
			FLOAT ux = f_1;
			ux += -f_3;
			ux +=  f_5;
			ux += -f_6;
			ux += -f_7;
			ux +=  f_8;
			// get contributions to y-velocities
			// these come from channels 2, 4, 5, 6, 7, 8
			/// 4,7,8 get a minus sign, 0,1,3 don't contribute
			FLOAT uy = f_2;
			uy += -f_4;
			uy +=  f_5;
			uy +=  f_6;
			uy += -f_7;
			uy += -f_8;
			// multiply with inverse density
			FLOAT rho_inv = 1./rho(x,y); // only divide once!
			ux *= rho_inv;
			uy *= rho_inv;
			// store the results
			vel(x,y,0) = ux;
			vel(x,y,1) = uy;
		}
	);
}

KOKKOS_INLINE_FUNCTION
FLOAT f_eq(FLOAT ux, FLOAT uy, FLOAT rho_i, UINT dir){
	/// @brief equilibrium weights of each discrete direction derived from a gaussian in 2D space
	/// (these values should sum up to one)
	const FLOAT w[9] = {
		4./9.,
		1./9.,
		1./9.,
		1./9.,
		1./9.,
		1./36.,
		1./36.,
		1./36.,
		1./36.,
	};
	FLOAT cx_ux = (dir==1 || dir==5 || dir==8) ? (ux) : ( 	// if dir is 1,5,8 then use cx=1
		(dir==3 || dir==6 || dir==7) ? (-ux) : ( 			// if dir is 3,6,7 then use cx=-1
			0.												// otherwise (0,2,4)		cx=0
		)
	);
	FLOAT cy_uy = (dir==2 || dir==5 || dir==6) ? (uy) : ( 	// if dir is 2,5,6 then use cy=1
		(dir==3 || dir==6 || dir==7) ? (-uy) : ( 			// if dir is 4,7,8 then use cy=-1
			0.												// otherwise (0,1,3)		cy=0
		)
	);
	FLOAT c_i_u_r = cx_ux + cy_uy;
	FLOAT u_2 = ux * ux + uy * uy;
	return w[dir] * rho_i * (
		1
		+ 3*c_i_u_r
		+ 9./2. * (c_i_u_r * c_i_u_r)
		- 3./2. * u_2
	);
}

void collide(Vel_t &vel, Den_t &rho, Dst_t &f, SFL &om){
	Kokkos::parallel_for(
		"collide", 
		Kokkos::MDRangePolicy({0, 0, 0}, {NX, NY, Q}), 
		KOKKOS_LAMBDA(const UINT x, const UINT y, const UINT dir){
			// load quantities into registers
			FLOAT f_i = VIEW(f, x,y,dir);
			FLOAT rho_i = rho(x,y);
			FLOAT ux = vel(x,y,0);
			FLOAT uy = vel(x,y,1);
			// interpolate towards the local equilibrium distribution
			const FLOAT OMEGA{om()};
			VIEW(f,x,y,dir) += OMEGA * (f_eq(ux, uy, rho_i, dir) - f_i);
		}
	);
}

/// @brief Set the distribution value at all nodes occupied by a wall to zero.
/// This is used after initialization to ensure all contributions that end up in walls
/// originate from neighbouring cells and can be bounced back
/// @param f the distribution field
/// @param wal conceptually a 1D field of 2 entries each describing the x and y-coordinate of a wall node
void clear_bdy(Dst_t &f, const Bdy_t &bdy){
	Kokkos::parallel_for(
		"set f at boundary to zero", 
		bdy.extent(0), 
		KOKKOS_LAMBDA(const UINT i){
			const UINT xi{bdy(i, 0)};
			const UINT yi{bdy(i, 1)};
			// set inital distribution to equilibrium distribution
			for (UINT dir{0}; dir<Q; ++dir){
				VIEW(f,xi,yi,dir) = 0.;
			}
		}
	);
}

void bounce_back_boundary(Dst_t &f, const Bdy_t &bdy, SUI &nx, SUI &ny){
	Kokkos::parallel_for(
		"implement bounce-back walls", 
		bdy.extent(0), 
		KOKKOS_LAMBDA(const UINT i){
			const UINT x{bdy(i, 0)};
			const UINT y{bdy(i, 1)};
			// read all required values
			const FLOAT f_1 { VIEW(f, x, y, 1) };
			const FLOAT f_2 { VIEW(f, x, y, 2) };
			const FLOAT f_3 { VIEW(f, x, y, 3) };
			const FLOAT f_4 { VIEW(f, x, y, 4) };
			const FLOAT f_5 { VIEW(f, x, y, 5) };
			const FLOAT f_6 { VIEW(f, x, y, 6) };
			const FLOAT f_7 { VIEW(f, x, y, 7) };
			const FLOAT f_8 { VIEW(f, x, y, 8) };
			// compute adjacent cell indices
			const UINT NX{nx()};
			const UINT NY{ny()};
			const UINT xr{(x+1 == NX) ? (0)    : (x+1)};
			const UINT xl{(x   == 0 ) ? (NX-1) : (x-1)};
			const UINT yu{(y+1 == NY) ? (0)    : (y+1)};
			const UINT yd{(y   == 0 ) ? (NY-1) : (y-1)};
			// bounce back contributions from the wall to the corresponding neighbouring cell
			// (no overwrites should occur here so this is inherently thread-safe)
			// 
			// 1 <-> 3     	| 6   2   5 |
			// 2 <-> 4     	|   \ | /   |
			// 5 <-> 7     	| 3 - 0 - 1 |
			// 6 <-> 8		|   / | \   |
			// 				| 7   4   8 |
			//
			// once again write in +y, then in +x-direction: 7 3 6 -> 4 0 2 -> 8 1 5
			VIEW(f, xl, yd, 7) = f_5;
			VIEW(f, xl, y , 3) = f_1;
			VIEW(f, xl, yu, 6) = f_8;
			VIEW(f, x , yd, 4) = f_2;
			// VIEW(f, x , y , 0) = f_0;
			VIEW(f, x , yu, 2) = f_4;
			VIEW(f, xr, yd, 8) = f_6;
			VIEW(f, xr, y , 1) = f_3;
			VIEW(f, xr, yu, 5) = f_7;
		}
	);
}

Bdy_t init_lid_driven(Vel_t &vel, Den_t &rho, Dst_t &f,  SUI &nx, SUI &ny){
	Bdy_t bdy("bdy", 2*NX+2*NY-4);
	// initialize boundary nodes
	Kokkos::parallel_for(
		"set f at boundary to zero", 
		bdy.extent(0), 
		KOKKOS_LAMBDA(const UINT i){
			const UINT NX{nx()};
			const UINT NY{ny()};

			if (i < NY){
				// left wall, bottom-up
				bdy(i,0) = 0;
				bdy(i,1) = i;
			} else if (i < 2*NY){
				// right wall, bottom-up
				bdy(i,0) = NX-1;
				bdy(i,1) = i -NY;
			} else if (i < 2*NY + NX - 2){
				// bottom wall, left-to-right
				bdy(i,0) = i -2*NY+1;
				bdy(i,1) = 0;
			} else {
				// top wall, left-to-right
				bdy(i,0) = i -(2*NY + NX - 2) +1;
				bdy(i,1) = NY-1;
			}
		}
	);

	// intialize f as equilibrium distribution of prescribed density and velocity fields
    Kokkos::parallel_for(
		"initialize", 
		Kokkos::MDRangePolicy({0, 0}, {NX, NY}), 
		KOKKOS_LAMBDA(const UINT x, const UINT y){
			// set initial densities
			FLOAT rho_i = RHO_INIT; // use initial densities
			rho(x,y) = rho_i;

			// set initial velocities
			const UINT NY{ny()};
			FLOAT ux{U_INIT * SIN((2.0*M_PI * ((FLOAT) y)) / ((FLOAT) NY))}; // this implements a shear wave
			FLOAT uy{0.};
			vel(x,y,0) = ux;
			vel(x,y,1) = uy;

			// set inital distribution to equilibrium distribution
			for (UINT dir{0}; dir<Q; ++dir){
				VIEW(f,x,y,dir) = f_eq(ux,uy,rho_i,dir);
			}
		}
	);

	// clear any distribution values that might have ended up in the boundary
	clear_bdy(f, bdy);
	return bdy;
}

void init_shearwave(Vel_t &vel, Den_t &rho, Dst_t &f, SUI &ny){
    Kokkos::parallel_for(
		"initialize", 
		Kokkos::MDRangePolicy({0, 0}, {NX, NY}), 
		KOKKOS_LAMBDA(const UINT x, const UINT y){
			// set initial densities
			FLOAT rho_i = RHO_INIT; // use initial densities
			rho(x,y) = rho_i;

			// set initial velocities
			const UINT NY{ny()};
			FLOAT ux{U_INIT * SIN((2.0*M_PI * ((FLOAT) y)) / ((FLOAT) NY))}; // this implements a shear wave
			FLOAT uy{0.};
			vel(x,y,0) = ux;
			vel(x,y,1) = uy;

			// set inital distribution to equilibrium distribution
			for (UINT dir{0}; dir<Q; ++dir){
				VIEW(f,x,y,dir) = f_eq(ux,uy,rho_i,dir);
			}
		}
	);
}

bool output(Vel_t &vel, Den_t &rho, Dst_t &f, Dst_t &buf){

	switch (OUTPUT)
	{
	case OUTPUT_TYPE::MAX_VEL:
		{
			// reconstruct density and velocity fields
			compute_densities(rho, f);
			compute_velocities(vel, rho, f);
			// find maximum velocity magnitude via parallel reduce
			FLOAT max_x_vel = -1e30;
			Kokkos::parallel_reduce(
				"find max x-velocity",
				Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {NX, NY}),
				KOKKOS_LAMBDA(const int i, const int j, FLOAT& local_max) {
					const FLOAT ux = vel(i, j, 0);
					const FLOAT uy = vel(i, j, 1);
					const FLOAT u = SQRT(ux*ux+uy*uy);
					if (u > local_max) local_max = u;
				},
				Kokkos::Max<FLOAT>(max_x_vel)
			);
			std::cout << max_x_vel << "," << std::flush;
		}
		break;
	case OUTPUT_TYPE::VEL_FIELD:
		{
			// reconstruct density and velocity fields
			compute_densities(rho, f);
			compute_velocities(vel, rho, f);
			// copy velocity from device to host-accessible buffer
			auto vel_host = Kokkos::create_mirror_view(vel);
			Kokkos::deep_copy(vel_host, vel);
			
			// print all velocities to std::out
			for (UINT y{0}; y<NY; ++y){
				for(UINT x{0}; x<NX; ++x){
					const FLOAT ux = vel_host(x,y,0);
					const FLOAT uy = vel_host(x,y,1);
					const FLOAT out = SQRT(ux*ux+uy*uy);
					if (out != out){
						// check for NaNs and abort
						std::cerr << "NaN encountered at x="<<x<<" y="<<y<< std::endl;
						return false;
					}
					// write to std::cout
					std::cout << out;
					if (x<NX-1){
						// add a comma for all but the last value
						std::cout << ",";
					}
				}
				std::cout << std::endl;
			}
			// seperate CSV blocks with "#" as a delimiter
			std::cout << "#" << std::endl;
		}
		break;
	default:
		break;
	}
	return true;
}

void parse_args(int argc, char *argv[]){
// define arguments to parse
	argparse::ArgumentParser program("lbm");
	program
		.add_argument("-nx", "--x-grid-points")
		.help("Specify the number of grid points in the x-direction")
		.default_value<UINT>({ 1024 })
  		.required()
		.store_into(NX);
	program
		.add_argument("-ny", "--y-grid-points")
		.help("Specify the number of grid points in the y-direction")
		.default_value<UINT>({ 1024 })
  		.required()
		.store_into(NY);
	program
		.add_argument("-of", "--output-frequency")
		.help("Specify how frequently (as an integer number of timesteps) simulation results should be output")
		.default_value<UINT>({ 100 })
  		.required()
		.store_into(OUT_EVERY_N);
	program
		.add_argument("-s", "--steps")
		.help("Specify the total number of simulation time steps to run")
		.default_value<UINT>({ 50000 })
  		.required()
		.store_into(STEPS);
	double omega;
	program
		.add_argument("-w", "--omega")
		.help("Specify the relaxation coefficient omega, which should obey be a float in (0;2)")
		.default_value<double>({ 0.5 })
  		.required()
		.store_into(omega);
	// mutually exclusive : the type of output
	auto &output_type = program.add_mutually_exclusive_group();
	output_type.add_argument("-omv", "--output-max-vel")
		.help("If specified, the program outputs the maximum velocity magnitude as a comma-seperated list of floats with a trailing comma")
		.implicit_value(true);
	output_type.add_argument("-ov", "--output-velocity")
		.help("If specified, the program outputs the velocity field as #-seperated CSV tables of NY columns and NX rows")
		.implicit_value(true);

	// parse the arguments
	try {
		program.parse_args(argc, argv);
	}
	catch (const std::exception& err) {
		std::cerr << err.what() << std::endl;
		std::cerr << program;
		std::exit(1);
	}


	// assign parsed arguments that are not already stored to corresponding variables
	OMEGA = (FLOAT) omega;
	if (program.is_used("-omv")){
		OUTPUT = OUTPUT_TYPE::MAX_VEL;
	} else if (program.is_used("-ov"))
	{
		OUTPUT = OUTPUT_TYPE::VEL_FIELD;
	}
}

bool run_simulation(SUI &nx, SUI &ny, SFL &om){
    Den_t rho("rho", NX, NY);
    Vel_t vel("vel", NX, NY);
	Dst_t f  ("f"  , NX, NY, Q);
    Dst_t buf("buf", NX, NY, Q);

	// initialize quantities
	// (buf need not be initialized since it is written to everywhere)
	init_shearwave(vel, rho, f, ny);
	// Bdy_t bdy{ init_lid_driven(vel, rho, f, nx, ny) };
	if (!output(vel, rho, f, buf)){ return false; };

    for (UINT t{0}; t < STEPS; ++t){
		// stream(f, buf, nx, ny);
		// compute_densities(rho, f);
		// compute_velocities(vel, rho, f);
		// collide(vel, rho, f, om);
		// push_periodic(f, buf, nx, ny, om);
		push_solid_boundary(f, buf, nx, ny, om);
		// bounce_back_boundary(f, bdy, nx, ny);

		// write results to std::out
		if (t % OUT_EVERY_N == 0){
			if (!output(vel, rho, f, buf)){ return false; };
		}
    };
	return true;
}


// MAIN
int main(int argc, char *argv[]) {
	// parse command line arguments to adjust NX, NY, OMEGA, ...
	parse_args(argc, argv);
	// Initialize Kokkos 
    Kokkos::initialize(argc, argv);
	// Kokkos::print_configuration(std::cout);
	// new scope to make sure deallocation precedes finalize()
	bool success = false;
	{
		// make args visible on device-side
		SUI nx("NX");
		SUI ny("NY");
		SFL om("OMEGA");
		auto host_nx = Kokkos::create_mirror_view(nx);
		auto host_ny = Kokkos::create_mirror_view(ny);
		auto host_om = Kokkos::create_mirror_view(om);
		host_nx() = NX;
		host_ny() = NY;
		host_om() = OMEGA;
		Kokkos::deep_copy(nx, host_nx);
		Kokkos::deep_copy(ny, host_ny);
		Kokkos::deep_copy(om, host_om);

		// Run the simulation
		success = run_simulation(nx, ny, om);
	}
    Kokkos::finalize();
    return success ? 0 : 1;
}
