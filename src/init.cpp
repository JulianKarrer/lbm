
#include "init.h"


void init_shearwave(Dst_t &f, DFL &rho_init, DFL &u, DUI &y_below, DUI &y_glob){
	// determine limits of the MDRange depending on whether a halo is used
	const UINT LX {USE_MPI ? (UINT)1 : (UINT)0};
	const UINT LY {USE_MPI ? (UINT)1 : (UINT)0};
	const UINT HX {USE_MPI ? NX-1 : NX};
	const UINT HY {USE_MPI ? NY-1 : NY};
	// pass indicator of whether to account for halo to device-side
	DUI use_mpi = create_device_uint(USE_MPI ? 1:0); 

    Kokkos::parallel_for(
		"initialize", 
		Kokkos::MDRangePolicy({LX, LY}, {HX, HY}, {TX, TY}), 
		KOKKOS_LAMBDA(const UINT x, const UINT y){
			const FLT rho_i { rho_init() };  // use initial density
			// set initial velocities
			const FLT u_init { u() }; 
			// scale y to rank-independent, global y in [0;1]
			const FLT y_scaled { 
				((FLT)(y - use_mpi() /*account for halo*/ + y_below() /*account for ranks below*/))
				/ ((FLT) (y_glob())) 
			}; 
			const FLT ux{u_init * SIN(2.0*M_PI * y_scaled)}; // this implements a shear wave

			// set inital distribution to equilibrium distribution
			for (UINT dir{0}; dir<Q; ++dir){
				VIEW(f,x,y,dir) = f_eq(ux,0.,rho_i,dir);
			}
		}
	);
}

void init_rest(Dst_t &f, DFL &rho_init){
    Kokkos::parallel_for(
		"initialize at rest", 
		Kokkos::MDRangePolicy({0, 0}, {NX, NY}, {TX, TY}), 
		KOKKOS_LAMBDA(const UINT x, const UINT y){
			// set initial densities to specified value
			const FLT rho_i { rho_init() }; 
			// the inital equilibrium distribution is therefore also zero
			for (UINT dir{0}; dir<Q; ++dir){
				VIEW(f,x,y,dir) = f_eq(0.,0.,rho_i,dir);
			}
		}
	);
}

/// @brief Compute the equilibrium distribution at the current node
/// @param ux x-component of the velocity
/// @param uy y-component of the velocity
/// @param rho_i current density at the node
/// @param dir channmel number of the direction under consideration
/// @return the equilibrium distribution along the requested channel
KOKKOS_INLINE_FUNCTION
FLT f_eq(FLT ux, FLT uy, FLT rho_i, UINT dir){
	/// @brief equilibrium weights of each discrete direction derived from a gaussian in 2D space
	/// (these values should sum up to one)
	constexpr FLT w[9] = {4./9.,1./9.,1./9.,1./9.,1./9.,1./36.,1./36.,1./36.,1./36.,};
	constexpr FLT cx[9] { 0, 1, 0,-1, 0, 1,-1,-1, 1};
	constexpr FLT cy[9] { 0, 0, 1, 0,-1, 1, 1,-1,-1};
	const FLT cx_ux{cx[dir] * ux};
	const FLT cy_uy{cy[dir] * uy};
	// FLT cx_ux = (dir==1 || dir==5 || dir==8) ? (ux) : ( 	// if dir is 1,5,8 then use cx=1
	// 	(dir==3 || dir==6 || dir==7) ? (-ux) : ( 			// if dir is 3,6,7 then use cx=-1
	// 		0.												// otherwise (0,2,4)		cx=0
	// 	)
	// );
	// FLT cy_uy = (dir==2 || dir==5 || dir==6) ? (uy) : ( 	// if dir is 2,5,6 then use cy=1
	// 	(dir==3 || dir==6 || dir==7) ? (-uy) : ( 			// if dir is 4,7,8 then use cy=-1
	// 		0.												// otherwise (0,1,3)		cy=0
	// 	)
	// );
	const FLT c_i_u_r = cx_ux + cy_uy;
	const FLT u_2 = ux * ux + uy * uy;
	return w[dir] * rho_i * (
		1
		+ 3*c_i_u_r
		+ 9./2. * (c_i_u_r * c_i_u_r)
		- 3./2. * u_2
	);
}