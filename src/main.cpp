#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <chrono>
#include "Kokkos_Core.hpp"
#include "macros.h"
#include "global.h"
#include "io.h"
#include "init.h"
#include "pull-push.h"

// defaults for macros in macros.h to avoid leaving them undefined
#ifndef USE_SINGLE_PRECISION
#define USE_SINGLE_PRECISION true
#endif
#ifndef UNROLL_LOOPS
#define UNROLL_LOOPS true
#endif
#ifndef COALESCING
#define COALESCING true
#endif
#ifndef USE_MPI
#define USE_MPI false
#endif

#if USE_MPI
#include <mpi.h>
#endif


bool run_simulation_single_node(){
	// create two buffers for use in two-grid, one-step updates
	#if !COALESCING
		Dst_t f  ("f"  , NY, NX, Q); // only for reading
		Dst_t buf("buf", NY, NX, Q); // only for writing
	#else
		Dst_t f  ("f"  , Q, NY, NX); // only for reading
		Dst_t buf("buf", Q, NY, NX); // only for writing
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

#if USE_MPI
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
			DUI y_glob = create_device_uint(NY_glob);		  // global y size: how many nodes exist globally in y-direction?
			init_shearwave(f, rho_init, u, y_below, y_glob); break;
	}

	// main simulation loop
    for (UINT t{0}; t < STEPS; ++t){
		// COMMUNICATE:
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
#endif


// MAIN ENTRY POINT

int main(int argc, char *argv[]) {
	#if USE_MPI
	// Initialize MPI if applicable
    MPI_Init(&argc, &argv);
	#endif

	// parse command line arguments to adjust NX, NY, OMEGA, ...
	parse_args(argc, argv);

	// Initialize Kokkos 
    Kokkos::initialize(argc, argv);
	// Kokkos::print_configuration(std::cout);

	// Run the simulation
	auto start {std::chrono::high_resolution_clock::now()};

	#if USE_MPI
		bool success = run_simulation_mpi();
	#else 
		bool success = run_simulation_single_node();
	#endif
	auto end {std::chrono::high_resolution_clock::now()};
	// print the MLUPS count
	auto span {std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count()};
	auto mlups = (NX*NY*STEPS*1000) / span;
	std::cerr << std::endl 
		<< NX*NY*STEPS << " lattice updates in |" 
		<< span << "|ns => " 
		<< mlups << " MLUPS"
		<< std::endl;
	(*OUT_STREAM) << std::flush;
    Kokkos::finalize();
	#if USE_MPI
	// Finalize MPI if applicable
    MPI_Finalize();
	#endif
    return success ? 0 : 1;
}
