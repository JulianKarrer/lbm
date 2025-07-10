
#include "pull-push.h"

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
