
#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <chrono>
#include "argparse/argparse.hpp"
#include "Kokkos_Core.hpp"


// TYPES

// define the used types and corresponding special functions as macros so they can be changed 
// (e.g. to benchmark double vs. float performance) 
// without refactoring the entire code
#define FLOAT float
#define SIN sinf
#define SQRT sqrtf
#define INT int32_t
#define UINT unsigned long long
// define an accessor macro for fields so that their memory layout 
// can be changed without refactoring every single function
#define VIEW(view, x, y, q) view(q, y, x)


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
OUTPUT_TYPE OUTPUT {OUTPUT_TYPE::NONE};

enum SCENE_TYPE{
	/// @brief simulate a shear-wave decay with x-velocities and periodic boundaries
	SHEAR_WAVE,
	/// @brief simulate a lid driven cavity with bounce-back solid walls
	LID_DRIVEN,
};
/// @brief the type of simulation to conduct (boundary and initial conditions)
SCENE_TYPE SCENE {SCENE_TYPE::SHEAR_WAVE};

/// @brief the stream to print output to. defaults to std::cout but may be overwritten with a file stream
std::ostream* OUT_STREAM{&std::cout};
/// @brief the file to write to, if specified and opened - otherwise, this is uninitialized! 
/// OUT_STREAM relies on this to have a static lifetime so the stream is not destructed before the program terminates.
std::ofstream OUT_FILE;

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
FLOAT OMEGA{1.7};
/// @brief initial density
FLOAT RHO_INIT{1.0};
/// @brief initial distribution values
FLOAT U_INIT{0.1};


// FIELD TYPES

/// Field of density values (x,y)
using Den_t = Kokkos::View<FLOAT**, Kokkos::LayoutRight>;

/// Field of velocity values (dir,x,y)
using Vel_t = Kokkos::View<FLOAT**[2], Kokkos::LayoutRight>;

/// Host-side field of velocity values (dir,x,y)
using Vel_t_host = Kokkos::View<FLOAT**[2], Kokkos::LayoutRight, Kokkos::HostSpace::device_type, Kokkos::Experimental::DefaultViewHooks>;

/// Field of distribution values
using Dst_t = Kokkos::View<FLOAT***, Kokkos::LayoutRight>; // Q Y X

/// Field of coordinates of bounce-back, no-slip walls
using Bdy_t = Kokkos::View<UINT*[2], Kokkos::LayoutRight>;


/// @brief A scalar, unsigned int
using SUI = Kokkos::View<UINT>;

/// @brief A scalar float
using SFL = Kokkos::View<FLOAT>;


// FUNCTIONS

/// @brief Use a one-step, two-grid push-scheme to access memory in f, compute a new pdf value locally (collision) 
/// and then write write to neighbours (streaming) in memory-order. 
/// 
/// -> Wittmann, Zeiser, Hager, Wellein "Comparison of different propagation steps for lattice Boltzmann methods"
/// @param f field holding distribution PDF values
/// @param buf temporary write-only-buffer with dimensions equal to f that gets pointer-swapped at the end
void push_periodic(Dst_t& f, Dst_t& buf, SUI &nx, SUI &ny, SFL &om){
	Kokkos::parallel_for(
		"push periodic", 
		Kokkos::MDRangePolicy<Kokkos::Rank<2, Kokkos::Iterate::Left, Kokkos::Iterate::Left>>({0, 0}, {NX, NY}), 
		KOKKOS_LAMBDA(const UINT x, const UINT y){

			// ###### READ VALUES #############################################
			const FLOAT f_0 { VIEW(f, x, y, 0) };
			const FLOAT f_1 { VIEW(f, x, y, 1) };
			const FLOAT f_2 { VIEW(f, x, y, 2) };
			const FLOAT f_3 { VIEW(f, x, y, 3) };
			const FLOAT f_4 { VIEW(f, x, y, 4) };
			const FLOAT f_5 { VIEW(f, x, y, 5) };
			const FLOAT f_6 { VIEW(f, x, y, 6) };
			const FLOAT f_7 { VIEW(f, x, y, 7) };
			const FLOAT f_8 { VIEW(f, x, y, 8) };
			// load other required scalar values
			const UINT NX{nx()};
			const UINT NY{ny()};
			const FLOAT OMEGA{om()};

			// ###### COMPUTE DENSITIES #######################################
			// collect density contributions from all directions
			const FLOAT rho {f_0 + f_1 + f_2 + f_3 + f_4 + f_5 + f_6 + f_7 + f_8};
			const FLOAT rho_inv {1./rho};

			// ###### COMPUTE VELOCITIES ######################################
			// contributions to x-velocities come from channels 1, 3, 5, 6, 7, 8
			/// 3,6,7 get a minus sign, 0,2,4 don't contribute
		 	const FLOAT ux { (f_1 - f_3 + f_5 - f_6 - f_7 + f_8) * rho_inv};
			// contributions to y-velocities come from channels 2, 4, 5, 6, 7, 8
			/// 4,7,8 get a minus sign, 0,1,3 don't contribute
			const FLOAT uy {(f_2 - f_4 + f_5 + f_6 - f_7 - f_8) * rho_inv};

			// ###### COMPUTE EQUILIBIRUM DISTRIBUITON AND STREAM #############
			// What the compiler ought to already do:
			// - factor out common subexpressions
			// - avoid pointless arithmetic like *1, *0, use - instead of *(-1)
			// - use conditionals instead of %
			// Other optimizations:
			// - reorder writes so they access subsequent memory

			const UINT xr{(x+1 == NX) ? (0)    : (x+1)};
			const UINT xl{(x   == 0 ) ? (NX-1) : (x-1)};
			const UINT yu{(y+1 == NY) ? (0)    : (y+1)};
			const UINT yd{(y   == 0 ) ? (NY-1) : (y-1)};

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
	);
	auto temp = f;
    f = buf;
    buf = temp;
}

void push_lid_driven(Dst_t& f, Dst_t& buf, SUI &nx, SUI &ny, SFL &om, SFL &rho_eq, SFL &u_lid){
	Kokkos::parallel_for(
		"push interior", 
		Kokkos::MDRangePolicy<Kokkos::Rank<2, Kokkos::Iterate::Left, Kokkos::Iterate::Left>>({0, 0}, {NX, NY}),
		KOKKOS_LAMBDA(const UINT x, const UINT y){
			// load the distribution
			const FLOAT f_0 { VIEW(f, x, y, 0) };
			const FLOAT f_1 { VIEW(f, x, y, 1) };
			const FLOAT f_2 { VIEW(f, x, y, 2) };
			const FLOAT f_3 { VIEW(f, x, y, 3) };
			const FLOAT f_4 { VIEW(f, x, y, 4) };
			const FLOAT f_5 { VIEW(f, x, y, 5) };
			const FLOAT f_6 { VIEW(f, x, y, 6) };
			const FLOAT f_7 { VIEW(f, x, y, 7) };
			const FLOAT f_8 { VIEW(f, x, y, 8) };
			// load relevant global scalars
			const UINT NX{nx()};
			const UINT NY{ny()};
			const FLOAT OMEGA{om()};
			const FLOAT U_LID{u_lid()};
			const FLOAT RHO_EQ{rho_eq()};
			// calculate density and velocity
			const FLOAT rho { f_0 + f_1 + f_2 + f_3 + f_4 + f_5 + f_6 + f_7 + f_8 };
			const FLOAT rho_inv {1./rho};
		 	const FLOAT ux { (f_1 - f_3 + f_5 - f_6 - f_7 + f_8) * rho_inv };
			const FLOAT uy { (f_2 - f_4 + f_5 + f_6 - f_7 - f_8) * rho_inv };
			// calculate equilibirum values
			const FLOAT ux_2 {4.5 * ux * ux};
			const FLOAT uy_2 {4.5 * uy * uy};
			const FLOAT uymux {uy - ux};
			const FLOAT uymux_2 {4.5 * uymux * uymux};
			const FLOAT uxpuy {ux + uy};
			const FLOAT uxpuy_2 {4.5 * uxpuy * uxpuy};
			const FLOAT u_2_times_3_2 {1.5 * (ux * ux + uy * uy)};
			const FLOAT w_4_9  {rho * 4./9.};
			const FLOAT w_1_9  {rho * 1./9.};
			const FLOAT w_1_36 {rho * 1./36.};
			const FLOAT f_7_eq { f_7 + OMEGA * ((w_1_36 * (1- 3*uxpuy+ uxpuy_2- u_2_times_3_2)) - f_7) };
			const FLOAT f_3_eq { f_3 + OMEGA * ((w_1_9 * (1- 3*ux+ ux_2- u_2_times_3_2)) - f_3) };
			const FLOAT f_6_eq { f_6 + OMEGA * ((w_1_36 * (1+ 3*uymux+ uymux_2- u_2_times_3_2)) - f_6) };
			const FLOAT f_4_eq { f_4 + OMEGA * ((w_1_9 * (1- 3*uy+ uy_2- u_2_times_3_2)) - f_4) };
			const FLOAT f_0_eq { f_0 + OMEGA * ((w_4_9 * (1- u_2_times_3_2)) - f_0) };
			const FLOAT f_2_eq { f_2 + OMEGA * ((w_1_9 * (1+ 3*uy+ uy_2- u_2_times_3_2)) - f_2) };
			const FLOAT f_8_eq { f_8 + OMEGA * ((w_1_36 * (1- 3*uymux+ uymux_2- u_2_times_3_2)) - f_8) };
			const FLOAT f_1_eq { f_1 + OMEGA * ((w_1_9 * (1+ 3*ux+ ux_2- u_2_times_3_2)) - f_1) };
			const FLOAT f_5_eq { f_5 + OMEGA * ((w_1_36 * (1+ 3*uxpuy+ uxpuy_2- u_2_times_3_2)) - f_5) };

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
			// const FLOAT df = {-2.0 * (1./36.) * RHO_EQ * (U_LID * (-1.))/(1./3.)};
			const FLOAT df = { t ? (RHO_EQ * U_LID / 6.) : 0. };  
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
			const FLOAT f_1 { VIEW(f, x, y, 1) };
			const FLOAT f_2 { VIEW(f, x, y, 2) };
			const FLOAT f_3 { VIEW(f, x, y, 3) };
			const FLOAT f_4 { VIEW(f, x, y, 4) };
			const FLOAT f_5 { VIEW(f, x, y, 5) };
			const FLOAT f_6 { VIEW(f, x, y, 6) };
			const FLOAT f_7 { VIEW(f, x, y, 7) };
			const FLOAT f_8 { VIEW(f, x, y, 8) };

			// OPTIMIZATION:
			// - From here on, unroll the loop over discrete directions, since only 6/9 entries of c 
			//   contribute to each component of the velocity, the others are zero.
			// - Also, negation can then be used instead of generic multiplication with -1.0

			// get contributions to x-velocities
			// these come from channels 1, 3, 5, 6, 7, 8
			/// 3,6,7 get a minus sign, 0,2,4 don't contribute
			const FLOAT ux {f_1-f_3+f_5-f_6-f_7+f_8};
			// get contributions to y-velocities
			// these come from channels 2, 4, 5, 6, 7, 8
			/// 4,7,8 get a minus sign, 0,1,3 don't contribute
			const FLOAT uy {f_2-f_4+f_5+f_6-f_7-f_8};
			// multiply with inverse density
			const FLOAT rho_inv {1./rho(x,y)}; // only divide once!
			// store the results
			vel(x,y,0) = ux * rho_inv;
			vel(x,y,1) = uy * rho_inv;
		}
	);
}

KOKKOS_INLINE_FUNCTION
FLOAT f_eq(FLOAT ux, FLOAT uy, FLOAT rho_i, UINT dir){
	/// @brief equilibrium weights of each discrete direction derived from a gaussian in 2D space
	/// (these values should sum up to one)
	constexpr FLOAT w[9] = {
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

void init_shearwave(Vel_t &vel, Den_t &rho, Dst_t &f, SUI &ny, SFL &rho_init, SFL &u){
    Kokkos::parallel_for(
		"initialize", 
		Kokkos::MDRangePolicy({0, 0}, {NX, NY}), 
		KOKKOS_LAMBDA(const UINT x, const UINT y){
			// set initial densities
			const FLOAT rho_i { rho_init() };  // use initial densities
			rho(x,y) = rho_i;

			// set initial velocities
			const UINT NY{ny()};
			const FLOAT u_init { u() }; 
			const FLOAT ux{u_init * SIN((2.0*M_PI * ((FLOAT) y)) / ((FLOAT) NY))}; // this implements a shear wave
			const FLOAT uy{0.};
			vel(x,y,0) = ux;
			vel(x,y,1) = uy;

			// set inital distribution to equilibrium distribution
			for (UINT dir{0}; dir<Q; ++dir){
				VIEW(f,x,y,dir) = f_eq(ux,uy,rho_i,dir);
			}
		}
	);
}

void init_rest(Vel_t &vel, Den_t &rho, Dst_t &f, SFL &rho_init){
    Kokkos::parallel_for(
		"initialize at rest", 
		Kokkos::MDRangePolicy({0, 0}, {NX, NY}), 
		KOKKOS_LAMBDA(const UINT x, const UINT y){
			// set initial densities to specified value
			const FLOAT rho_i { rho_init() }; 
			rho(x,y) = rho_i;
			// set velocities to zero
			vel(x,y,0) = 0.;
			vel(x,y,1) = 0.;
			// the inital equilibrium distribution is therefore also zero
			for (UINT dir{0}; dir<Q; ++dir){
				VIEW(f,x,y,dir) = f_eq(0.,0.,rho_i,dir);
			}
		}
	);
}

bool output(Vel_t &vel, Den_t &rho, Dst_t &f, Dst_t &buf){
	// output depending on selected OUTPUT_TYPE
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
			(*OUT_STREAM) << max_x_vel << "," << std::flush;
		}
		break;
	case OUTPUT_TYPE::VEL_MAGS:
		{	
			// reconstruct density and velocity fields
			compute_densities(rho, f);
			compute_velocities(vel, rho, f);
			// copy velocity from device to host-accessible buffer
			auto vel_host {Kokkos::create_mirror_view(vel)};
			Kokkos::deep_copy(vel_host, vel);
			
			// print all velocities to output stream
			for (UINT y{0}; y<NY; ++y){
				for(UINT x{0}; x<NX; ++x){
					const FLOAT ux = VIEW(vel_host,x,y,0);
					const FLOAT uy = VIEW(vel_host,x,y,1);
					const FLOAT out = SQRT(ux*ux+uy*uy);
					if (out != out){
						// check for NaNs and abort
						std::cerr << "NaN encountered at x="<<x<<" y="<<y<< std::endl;
						return false;
					}
					// write to output stream
					(*OUT_STREAM) << out;
					if (x<NX-1){
						// add a comma for all but the last value
						(*OUT_STREAM) << ",";
					}
				}
				(*OUT_STREAM) << std::endl;
			}
			// seperate CSV blocks with "#" as a delimiter
			(*OUT_STREAM) << "#" << std::endl;
		}
		break;
	case OUTPUT_TYPE::VEL_FIELD:
		{	
			// reconstruct density and velocity fields
			compute_densities(rho, f);
			compute_velocities(vel, rho, f);
			// copy velocity from device to host-accessible buffer
			auto vel_host {Kokkos::create_mirror_view(vel)};
			Kokkos::deep_copy(vel_host, vel);
			
			// print all x-components
			for (UINT y{0}; y<NY; ++y){
				for(UINT x{0}; x<NX; ++x){
					const FLOAT ux = VIEW(vel_host,x,y,0);
					(*OUT_STREAM) << ux;
					if (x<NX-1){(*OUT_STREAM) << ",";}
				}
				(*OUT_STREAM) << std::endl;
			}
			(*OUT_STREAM) << "#" << std::endl;
			// print all y-components
			for (UINT y{0}; y<NY; ++y){
				for(UINT x{0}; x<NX; ++x){
					const FLOAT uy = VIEW(vel_host,x,y,1);
					(*OUT_STREAM) << uy;
					if (x<NX-1){(*OUT_STREAM) << ",";}
				}
				(*OUT_STREAM) << std::endl;
			}
			(*OUT_STREAM) << "#" << std::endl;

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
	double rho_init;
	program
		.add_argument("-rho", "--density")
		.help("Specify the initial density at each fluid node")
		.default_value<double>({ 1.0 })
  		.required()
		.store_into(rho_init);
	double u_init;
	program
		.add_argument("-u", "--initial-velocity")
		.help("Specify the initial velocity. For the shear wave decay, this is the velocity amplitude. For the lid-driven cavity, this is the horizontal velocity of the top lid wall.")
		.default_value<double>({ 0.1 })
  		.required()
		.store_into(u_init);
	// mutually exclusive : the type of output
	auto &output_type = program.add_mutually_exclusive_group();
	output_type.add_argument("-omv", "--output-max-vel")
		.help("If specified, the program outputs the maximum velocity magnitude as a comma-seperated list of floats with a trailing comma")
		.implicit_value(true);
	output_type.add_argument("-ov", "--output-velocity")
		.help("If specified, the program outputs the velocity magnitude at each node as #-seperated CSV tables of NY columns and NX rows")
		.implicit_value(true);
	output_type.add_argument("-ovf", "--output-velocity-field")
		.help("If specified, the program outputs the components of the velocity field at each node as #-seperated CSV tables of NY columns and NX rows, first all x-components, then all y-components")
		.implicit_value(true);
	// optionally write to file
	std::string filename;
		program.add_argument("-f", "--file")
		.help("Specify a filename to write to instead of printing output to std::cout")
		.store_into(filename);
	// determine the type of simulation
	auto &sim_type = program.add_mutually_exclusive_group();
	sim_type.add_argument("-sw", "--shear-wave-decay")
		.help("Simulate a shear-wave decay with velocities in x-direction and periodic boundaries. This is the default.")
		.implicit_value(true);
	sim_type.add_argument("-ld", "--lid-driven-cavity")
		.help("Simulate a lid driven cavity with bounce-back solid walls and a moving wall at the top, where the fluid is initially at rest.")
		.implicit_value(true);

	// parse the arguments
	try {
		program.parse_args(argc, argv);
		if (program.is_used("-f")){
			OUT_FILE.open(filename);
			if (!OUT_FILE.is_open()) {
                std::cerr << "Failed to open file: " << filename << "\n";
				std::exit(1);
            }
			OUT_STREAM = &OUT_FILE;
		}
	}
	catch (const std::exception& err) {
		std::cerr << err.what() << std::endl;
		std::cerr << program;
		std::exit(1);
	}

	// assign parsed arguments that are not already stored to corresponding variables
	// omega, rho_init, u_init
	OMEGA = (FLOAT) omega;
	RHO_INIT = (FLOAT) rho_init;
	U_INIT = (FLOAT) u_init;
	// the kind of output
	if (program.is_used("-omv")){
		OUTPUT = OUTPUT_TYPE::MAX_VEL;
	} else if (program.is_used("-ov")){
		OUTPUT = OUTPUT_TYPE::VEL_MAGS;
	}else if (program.is_used("-ovf")){
		OUTPUT = OUTPUT_TYPE::VEL_FIELD;
	}
	// the kind of boundary and initial conditions
	if (program.is_used("-ld")){
		SCENE = SCENE_TYPE::LID_DRIVEN;
	} else {
		SCENE = SCENE_TYPE::SHEAR_WAVE;
	}
}

bool run_simulation(SUI &nx, SUI &ny, SFL &om, SFL &rho_init, SFL &u){
    Den_t rho("rho", NX, NY);
    Vel_t vel("vel", NX, NY);
	Dst_t f  ("f"  , Q, NY, NX);
    Dst_t buf("buf", Q, NY, NX);

	// initialize quantities
	// (buf need not be initialized since it is written to everywhere)
	switch (SCENE){
		case SCENE_TYPE::LID_DRIVEN:
			// init_shearwave(vel, rho, f, ny, rho_init, u); break;
			init_rest(vel, rho, f, rho_init); break;
		default:
			init_shearwave(vel, rho, f, ny, rho_init, u); break;
	}

    for (UINT t{0}; t < STEPS; ++t){
		// do a single fused step of collision and streaming, tailored to the respective boundary conditions of the scene
		switch (SCENE){
			case SCENE_TYPE::LID_DRIVEN:
				push_lid_driven(f, buf, nx, ny, om, rho_init, u);
				// boundary_correct_lid(f, nx, ny, rho_init, u);
				break;
			default:
				push_periodic(f, buf, nx, ny, om); break;
		}
		// report progress
		if (OUT_EVERY_N > 0){
			std::cerr << t << " / " << STEPS << "\r";
		}
		// write results to std::out
		if (OUT_EVERY_N > 0 && t % OUT_EVERY_N == 0){
			if (!output(vel, rho, f, buf)){ return false; };
		}
    };

	return true;
}


// MAIN ENTRY POINT

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
		SFL rho("RHO_INIT");
		SFL u("U_INIT");
		auto host_nx = Kokkos::create_mirror_view(nx);
		auto host_ny = Kokkos::create_mirror_view(ny);
		auto host_om = Kokkos::create_mirror_view(om);
		auto host_rho = Kokkos::create_mirror_view(rho);
		auto host_u = Kokkos::create_mirror_view(u);
		host_nx() = NX;
		host_ny() = NY;
		host_om() = OMEGA;
		host_rho() = RHO_INIT;
		host_u() = U_INIT;
		Kokkos::deep_copy(nx, host_nx);
		Kokkos::deep_copy(ny, host_ny);
		Kokkos::deep_copy(om, host_om);
		Kokkos::deep_copy(u, host_u);
		Kokkos::deep_copy(rho, host_rho);

		// Run the simulation
		auto start {std::chrono::high_resolution_clock::now()};
		success = run_simulation(nx, ny, om, rho, u);
		auto end {std::chrono::high_resolution_clock::now()};
		// print the MLUPS count
		auto span {std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count()};
		auto mlups = (NX*NY*STEPS*1000) / span;
		std::cerr << std::endl 
			<< NX*NY*STEPS << " lattice updates in " 
			<< span*1e-9 << "s => " 
			<< mlups << " MLUPS"
			<< std::endl;
	}
	(*OUT_STREAM) << std::flush;
    Kokkos::finalize();
    return success ? 0 : 1;
}
