
from math import sqrt
import subprocess
from datetime import datetime

NX = 32_000
NY = 32_000
STEPS = 100 # (32k)^2, T=100 => 16s
REPEATS=5

UNROLL_LOOPS=True
USE_SINGLE_PRECISION=True
CONTIGUOUS = True
USE_MPI = False

def compile():
    subprocess.run(["cmake", "--build", "build", "--parallel"], check=True)

def measure_mlups(args=""):
    command = f"mpirun --bind-to none --map-by slot -np 1 ./main -nx {NX} -ny {NY} -of {0} -s {STEPS} --shear-wave-decay"+args
    print("Command running:", command)
    # run the command repeatedly, collecting MLUPS measurements
    results = []
    for _ in range(REPEATS):
        output = subprocess.run(command.split(" "), cwd="build", stdout=subprocess.PIPE,stderr=subprocess.PIPE, text=True, check=False)
        print(output.stderr)
        ns = int(output.stderr.splitlines()[-1].split("|")[-2])
        results += [NX*NY*STEPS*1_000/ns]
    # compute average and standard deviation
    average = sum(results)/REPEATS
    sample_standard_dev = sqrt(sum([(r - average)**2 for r in results])/(REPEATS-1))
    return average, sample_standard_dev

def overwrite_macros():
    with open("./src/macros.h","w") as f:
        f.write(f"""
#ifndef MACROS_H
#define MACROS_H

#define USE_SINGLE_PRECISION {"true" if USE_SINGLE_PRECISION else "false"}
#define UNROLL_LOOP {"true" if UNROLL_LOOPS else "false"}
#define COALESCING {"true" if CONTIGUOUS else "false"}
#define USE_MPI {"true" if USE_MPI else "false"}

#endif""")

if __name__=="__main__":

    # normal run
    overwrite_macros()
    compile()
    pull_unrolled, pull_unrolled_stddev = measure_mlups()
    push_unrolled, push_unrolled_stddev = measure_mlups(" -push")

    # run with non-contiguous reads/writes
    CONTIGUOUS = False
    overwrite_macros()
    compile()
    pull_uncont, pull_uncont_stddev = measure_mlups()
    push_uncont, push_uncont_stddev = measure_mlups(" -push")
    CONTIGUOUS = True

    # run with non-unrolled loops
    UNROLL_LOOPS = False
    overwrite_macros()
    compile()
    pull_rolled, pull_rolled_stddev = measure_mlups()
    push_rolled, push_rolled_stddev = measure_mlups(" -push")
    UNROLL_LOOPS = True

    # run with double precision
    USE_SINGLE_PRECISION = False
    overwrite_macros()
    compile()
    pull_double, pull_double_stddev = measure_mlups()
    push_double, push_double_stddev = measure_mlups(" -push")

    # reset macros
    UNROLL_LOOPS = True
    USE_SINGLE_PRECISION = True
    CONTIGUOUS = True
    overwrite_macros()

    # write out results
    name = datetime.now().strftime("%Y_%m_%d_%H:%M:%S")
    with open(f"benchmark{name}.results","w") as f:
        f.write(f"""
pull_unrolled   {pull_unrolled} +- {pull_unrolled_stddev}
push_unrolled   {push_unrolled} +- {push_unrolled_stddev}
pull_uncont     {pull_uncont}   +- {pull_uncont_stddev}
push_uncont     {push_uncont}   +- {push_uncont_stddev}
pull_rolled     {pull_rolled}   +- {pull_rolled_stddev}
push_rolled     {push_rolled}   +- {push_rolled_stddev}
pull_double     {pull_double}   +- {pull_double_stddev}
push_double     {push_double}   +- {push_double_stddev}
""")

