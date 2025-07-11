#ifndef IO_H
#define IO_H
#include "global.h"
#include "argparse/argparse.hpp"

#if USE_MPI
#include <mpi.h>
#endif

/// @brief Parse command line arguments, storing values into the corresponding global variables on the host-side.
/// Arguments used on the device-side must be moved into buffers subsequently.
/// @param argc argument count
/// @param argv argument char pointer
void parse_args(int argc, char *argv[]);

/// @brief Output results of the type `OUTPUT_TYPE` to the output stream `OUT_STREAM`. Both of these are global variables set by `arg_parse` and defined in `types.h`
/// @param vel A buffer for reconstructing the velocity field
/// @param f the distribution 
/// @param LX lower limit of x-coordinates to output
/// @param LY lower limit of y-coordinates to output
/// @param HX upper limit of x-coordinates to output
/// @param HY upper limit of y-coordinates to output
/// @param rank own MPI rank - this is ignored if not applicable (`USEMPI==false`)
/// @return 
bool output(Vel_t &vel, Dst_t &f, UINT LX, UINT LY, UINT HX, UINT HY, int rank);

#endif