#include "io.h"

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
		.help("Specify how frequently (as an integer number of timesteps) simulation results should be output. 0 means no output.")
		.default_value<UINT>({ 0 })
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
	program
		.add_argument("-tx", "--x-grid-points")
		.help("Specify the kernel tiling size in x-direction for performance optimization")
		.default_value<UINT>({ 512 })
  		.required()
		.store_into(TX);
	program
		.add_argument("-ty", "--y-grid-points")
		.help("Specify the kernel tiling size in y-direction for performance optimization")
		.default_value<UINT>({ 1 })
  		.required()
		.store_into(TY);
	program
		.add_argument("-push", "--push")
		.help("Specify whether to prefer a push-type streaming pattern over pull-type. Pulling is used by default.")
		.implicit_value(true);
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
	OMEGA = (FLT) omega;
	RHO_INIT = (FLT) rho_init;
	U_INIT = (FLT) u_init;
	// the kind of output
	if (program.is_used("-omv")){
		OUTPUT = OUTPUT_TYPE::MAX_VEL;
	} else if (program.is_used("-ov")){
		OUTPUT = OUTPUT_TYPE::VEL_MAGS;
	} else if (program.is_used("-ovf")){
		OUTPUT = OUTPUT_TYPE::VEL_FIELD;
	}
	// whether to use push kernels
	if (program.is_used("-push")){
		PUSH = true;
	}
	// the kind of boundary and initial conditions
	if (program.is_used("-ld")){
		SCENE = SCENE_TYPE::LID_DRIVEN;
	} else {
		SCENE = SCENE_TYPE::SHEAR_WAVE;
	}
}


void compute_velocities(Vel_t &vel, Dst_t &f, UINT LX, UINT LY, UINT HX, UINT HY){
	Kokkos::parallel_for(
		"compute velocities", 
		Kokkos::MDRangePolicy({LX, LY}, {HX, HY}, {TX, TY}), 
		KOKKOS_LAMBDA(const UINT x, const UINT y){
			// load f_i values into registers to avoid coalescing requirements for multiple accesses
			// -> note that f_0 is not loaded since it would be weighted with a zero velocity anyways
			const FLT f_0 { VIEW(f, x, y, 0) };
			const FLT f_1 { VIEW(f, x, y, 1) };
			const FLT f_2 { VIEW(f, x, y, 2) };
			const FLT f_3 { VIEW(f, x, y, 3) };
			const FLT f_4 { VIEW(f, x, y, 4) };
			const FLT f_5 { VIEW(f, x, y, 5) };
			const FLT f_6 { VIEW(f, x, y, 6) };
			const FLT f_7 { VIEW(f, x, y, 7) };
			const FLT f_8 { VIEW(f, x, y, 8) };

			// OPTIMIZATION:
			// - From here on, unroll the loop over discrete directions, since only 6/9 entries of c 
			//   contribute to each component of the velocity, the others are zero.
			// - Also, negation can then be used instead of generic multiplication with -1.0

			// get contributions to x-velocities
			// these come from channels 1, 3, 5, 6, 7, 8
			/// 3,6,7 get a minus sign, 0,2,4 don't contribute
			const FLT ux {f_1-f_3+f_5-f_6-f_7+f_8};
			// get contributions to y-velocities
			// these come from channels 2, 4, 5, 6, 7, 8
			/// 4,7,8 get a minus sign, 0,1,3 don't contribute
			const FLT uy {f_2-f_4+f_5+f_6-f_7-f_8};
			// multiply with inverse density
			constexpr FLT f1 {1.0};
			const FLT rho_inv {f1/(f_0 + f_1 + f_2 + f_3 + f_4 + f_5 + f_6 + f_7 + f_8)}; // only divide once!
			// store the results
			vel(x,y,0) = ux * rho_inv;
			vel(x,y,1) = uy * rho_inv;
		}
	);
}


bool output(Vel_t &vel, Dst_t &f, UINT LX, UINT LY, UINT HX, UINT HY, int rank){
	// output depending on selected OUTPUT_TYPE
	switch (OUTPUT)
	{
	case OUTPUT_TYPE::MAX_VEL:
		{
			// find maximum velocity magnitude via parallel reduce
			FLT max_vel_mag = -1e30;
			Kokkos::parallel_reduce(
				"find max x-velocity",
				Kokkos::MDRangePolicy<Kokkos::Rank<2>>({LX, LY}, {HX, HY}),
				KOKKOS_LAMBDA(const int x, const int y, FLT& local_max) {
					// compute velocity
					const FLT f_0 { VIEW(f, x, y, 0) };
					const FLT f_1 { VIEW(f, x, y, 1) };
					const FLT f_2 { VIEW(f, x, y, 2) };
					const FLT f_3 { VIEW(f, x, y, 3) };
					const FLT f_4 { VIEW(f, x, y, 4) };
					const FLT f_5 { VIEW(f, x, y, 5) };
					const FLT f_6 { VIEW(f, x, y, 6) };
					const FLT f_7 { VIEW(f, x, y, 7) };
					const FLT f_8 { VIEW(f, x, y, 8) };
					constexpr FLT f1 {1.0};
					const FLT rho_inv {f1/(f_0 + f_1 + f_2 + f_3 + f_4 + f_5 + f_6 + f_7 + f_8)};
					const FLT ux {(f_1-f_3+f_5-f_6-f_7+f_8)*rho_inv};
					const FLT uy {(f_2-f_4+f_5+f_6-f_7-f_8)*rho_inv};
					// compute magnitude and perform reduction
					const FLT u = SQRT(ux*ux+uy*uy);
					if (u > local_max) local_max = u;
				},
				Kokkos::Max<FLT>(max_vel_mag)
			);

			#if USE_MPI
				// MPI: max-reduce the local maximum velocity magnitude, output on rank 0
				FLT global_max {max_vel_mag};
				MPI_Reduce(&max_vel_mag, &global_max, 1, MFLOAT, MPI_MAX, 0, MPI_COMM_WORLD);
				if (rank==0){
					(*OUT_STREAM) << global_max << "," << std::flush;
				}
			#else 
				// single node: just output the maximum velocity magnitude
				(*OUT_STREAM) << max_vel_mag << "," << std::flush;
			#endif
		}
		break;
	case OUTPUT_TYPE::VEL_MAGS:
		{	
			// TODO:
			if (USE_MPI){std::cerr<<"This output type is currently unsupported in MPI mode. Try --nmpi."<<std::endl;return false;}
			// reconstruct density and velocity fields
			compute_velocities(vel, f, LX, LY, HX, HY);
			// copy velocity from device to host-accessible buffer
			auto vel_host {Kokkos::create_mirror_view(vel)};
			Kokkos::deep_copy(vel_host, vel);
			
			// print all velocities to output stream
			for (UINT y{LY}; y<HY; ++y){
				for(UINT x{LX}; x<HX; ++x){
					const FLT ux = vel_host(x,y,0);
					const FLT uy = vel_host(x,y,1);
					const FLT out = SQRT(ux*ux+uy*uy);
					if (out != out){
						// check for NaNs and abort
						std::cerr << "NaN encountered at x="<<x<<" y="<<y<< std::endl;
						return false;
					}
					// write to output stream
					(*OUT_STREAM) << out;
					if (x<HX-1){
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
			// TODO:
			if (USE_MPI){std::cerr<<"This output type is currently unsupported in MPI mode. Try --nmpi."<<std::endl;return false;}
			// reconstruct density and velocity fields
			compute_velocities(vel, f, LX, LY, HX, HY);
			// copy velocity from device to host-accessible buffer
			auto vel_host {Kokkos::create_mirror_view(vel)};
			Kokkos::deep_copy(vel_host, vel);
			
			// print all x-components
			for (UINT y{LY}; y<HY; ++y){
				for(UINT x{LX}; x<HX; ++x){
					const FLT ux = vel_host(x,y,0);
					(*OUT_STREAM) << ux;
					if (x<HX-1){(*OUT_STREAM) << ",";}
				}
				(*OUT_STREAM) << std::endl;
			}
			(*OUT_STREAM) << "#" << std::endl;
			// print all y-components
			for (UINT y{LY}; y<HY; ++y){
				for(UINT x{LX}; x<HX; ++x){
					const FLT uy = vel_host(x,y,1);
					(*OUT_STREAM) << uy;
					if (x<HX-1){(*OUT_STREAM) << ",";}
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

