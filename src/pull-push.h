#ifndef PULL_PUSH_H
#define PULL_PUSH_H

#include "global.h"

KOKKOS_INLINE_FUNCTION
/// @brief Apply a fused Stream+Collide using the Pull scheme (read from neighbours, write locally) at the specified coordinates.
/// This is the manually loop-unrolled and hand-optimized version of the code.
/// @param x x-coordinate
/// @param y y-coordinate
/// @param xl x-coordinate of left neighbour
/// @param xr x-coordinate of right neighbour
/// @param yd y-coordinate of lower neighbour
/// @param yu y-coordinate of upper neighbour
/// @param f distribution values (read)
/// @param buf distribution values (write)
/// @param OMEGA relaxation parameter ω
void pull_periodic(const UINT x, const UINT y, const UINT xl, const UINT xr, const UINT yd, const UINT yu, Dst_t const& f, Dst_t const& buf, FLT OMEGA);

KOKKOS_INLINE_FUNCTION
/// @brief Apply a fused Stream+Collide using the Pull scheme (read from neighbours, write locally) at the specified coordinates
/// This is the NOT hand-optimized version of the code.
/// @param x x-coordinate
/// @param y y-coordinate
/// @param xl x-coordinate of left neighbour
/// @param xr x-coordinate of right neighbour
/// @param yd y-coordinate of lower neighbour
/// @param yu y-coordinate of upper neighbour
/// @param f distribution values (read)
/// @param buf distribution values (write)
/// @param OMEGA relaxation parameter ω
void pull_periodic_for(const UINT x, const UINT y, const UINT xl, const UINT xr, const UINT yd, const UINT yu, Dst_t const& f_i, Dst_t const& buf, FLT OMEGA);

KOKKOS_INLINE_FUNCTION
/// @brief Apply a fused Stream+Collide using the Push scheme (read locally, write to neighbours) at the specified coordinates
/// This is the hand-optimized version of the code.
/// @param x x-coordinate
/// @param y y-coordinate
/// @param xl x-coordinate of left neighbour
/// @param xr x-coordinate of right neighbour
/// @param yd y-coordinate of lower neighbour
/// @param yu y-coordinate of upper neighbour
/// @param f distribution values (read)
/// @param buf distribution values (write)
/// @param OMEGA relaxation parameter ω
void push_periodic(const UINT x, const UINT y, const UINT xl, const UINT xr, const UINT yd, const UINT yu, Dst_t const& f, Dst_t const& buf, FLT OMEGA);


KOKKOS_INLINE_FUNCTION
/// @brief Apply a fused Stream+Collide using the Push scheme (read locally, write to neighbours) at the specified coordinates
/// This is the NOT hand-optimized version of the code.
/// @param x x-coordinate
/// @param y y-coordinate
/// @param xl x-coordinate of left neighbour
/// @param xr x-coordinate of right neighbour
/// @param yd y-coordinate of lower neighbour
/// @param yu y-coordinate of upper neighbour
/// @param f distribution values (read)
/// @param buf distribution values (write)
/// @param OMEGA relaxation parameter ω
void push_periodic_for(const UINT x, const UINT y, const UINT xl, const UINT xr, const UINT yd, const UINT yu, Dst_t const& f_i, Dst_t const& buf, FLT OMEGA);


/// @brief Use a one-step, two-grid push-scheme to access memory in f, compute a new pdf value locally (collision) 
/// and then write write to neighbours (streaming) in memory-order. 
/// 
/// -> Wittmann, Zeiser, Hager, Wellein "Comparison of different propagation steps for lattice Boltzmann methods"
/// @param f field holding distribution PDF values
/// @param buf temporary write-only-buffer with dimensions equal to f that gets pointer-swapped at the end
/// @param nx total rank-local number of nodes in x-direction (device-side)
/// @param ny total rank-local number of nodes in y-direction (device-side)
/// @param om relaxation parameter ω (device-side)
void push_periodic_no_mpi(Dst_t& f, Dst_t& buf, DUI &nx, DUI &ny, DFL &om);


/// @brief Use a one-step, two-grid pull-scheme to access memory in f, read values from neighbours (streaming), compute a new pdf value (collision) and then write it at the local node.
/// 
/// -> Wittmann, Zeiser, Hager, Wellein "Comparison of different propagation steps for lattice Boltzmann methods"
/// @param f field holding distribution PDF values
/// @param buf temporary write-only-buffer with dimensions equal to f that gets pointer-swapped at the end
/// @param nx total rank-local number of nodes in x-direction (device-side)
/// @param ny total rank-local number of nodes in y-direction (device-side)
/// @param om relaxation parameter ω (device-side)
void pull_periodic_no_mpi(Dst_t& f, Dst_t& buf, DUI &nx, DUI &ny, DFL &om);


/// @brief Update the inner nodes, not reliant on the halo region, using a pull scheme
/// @param f distribution values (read)
/// @param buf distribution values (write)
/// @param om relaxation parameter ω (device-side)
/// @param LX lower bound of x-coords (inclusive)
/// @param LY lower bound of y-coords (inclusive)
/// @param HX upper bound of x-coords (exclusive)
/// @param HY upper bound of y-coords (exclusive)
void pull_periodic_inner(Dst_t& f, Dst_t& buf, DFL &om, UINT LX, UINT LY, UINT HX, UINT HY);



/// @brief Update the inner nodes, not reliant on the halo region, using a push scheme
/// @param f distribution values (read)
/// @param buf distribution values (write)
/// @param om relaxation parameter ω (device-side)
/// @param LX lower bound of x-coords (inclusive)
/// @param LY lower bound of y-coords (inclusive)
/// @param HX upper bound of x-coords (exclusive)
/// @param HY upper bound of y-coords (exclusive)
void push_periodic_inner(Dst_t& f, Dst_t& buf, DFL &om, UINT LX, UINT LY, UINT HX, UINT HY);



/// @brief Update the outer nodes, the ones reliant on the halo region, using a pull scheme
/// @param f distribution values (read)
/// @param buf distribution values (write)
/// @param om relaxation parameter ω (device-side)
/// @param nx total rank-local number of nodes in x-direction (device-side)
/// @param ny total rank-local number of nodes in y-direction (device-side)
void pull_periodic_outer(Dst_t& f, Dst_t& buf, DFL &om, DUI &nx, DUI &ny);


/// @brief Update the outer nodes, the ones reliant on the halo region, using a push scheme
/// @param f distribution values (read)
/// @param buf distribution values (write)
/// @param om relaxation parameter ω (device-side)
/// @param nx total rank-local number of nodes in x-direction (device-side)
/// @param ny total rank-local number of nodes in y-direction (device-side)
void push_periodic_outer(Dst_t& f, Dst_t& buf, DFL &om, DUI &nx, DUI &ny);


/// @brief Use a one-step, two-grid push-scheme to access memory in f, compute a new pdf value locally (collision) 
/// and then write to neighbours (streaming). 
/// -> Wittmann, Zeiser, Hager, Wellein "Comparison of different propagation steps for lattice Boltzmann methods"
/// - THIS version has incorporated boundary conditions for a single-rank lid driven cavity, where the top wall moves in x-direction and all  other walls enforce bounce-back bounaries.
/// @param f field holding distribution PDF values
/// @param buf temporary write-only-buffer with dimensions equal to f that gets pointer-swapped at the end
/// @param nx total rank-local number of nodes in x-direction (device-side)
/// @param ny total rank-local number of nodes in y-direction (device-side)
/// @param om relaxation parameter ω (device-side)
/// @param rho_eq rest density ρ (device-side)
/// @param u_lid velocity of the sliding lid (device-side)
void push_lid_driven(Dst_t& f, Dst_t& buf, DUI &nx, DUI &ny, DFL &om, DFL &rho_eq, DFL &u_lid);

#endif