#ifndef INIT_H
#define INIT_H

#include "global.h"
#include "Kokkos_Core.hpp"

/// @brief Initialize a shearwave decay scenario with a full 2π sine wave profile of varying x-velcotities along the y axis. 
/// @param f distribution buffer
/// @param rho_init initial density
/// @param u initial velocity amplitude (maximum u_x)
/// @param y_below number of nodes below the y=0 coordinate of the current rank, useful for initialization in an MPI setting (otherwise set to zero)
/// @param y_glob number of nodes along the y-axis globally, across all MPI ranks (for one rank, this should be NY)
void init_shearwave(Dst_t &f, DFL &rho_init, DFL &u, DUI &y_below, DUI &y_glob);

/// @brief Initialize a scenario where the fluid is at rest with a given density. This may be used with lid-driven cavity boundary conditions.
/// @param f distribution buffer
/// @param rho_init initial density
void init_rest(Dst_t &f, DFL &rho_init);

/// @brief Compute the equilibrium distribution at the current node
/// @param ux x-component of the velocity
/// @param uy y-component of the velocity
/// @param rho_i current density at the node
/// @param dir channmel number of the direction under consideration
/// @return the equilibrium distribution along the requested channel
KOKKOS_INLINE_FUNCTION
FLT f_eq(FLT ux, FLT uy, FLT rho_i, UINT dir);

#endif