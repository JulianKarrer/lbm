
from math import sqrt
import subprocess
from datetime import datetime

NX = 32_000
NY = 32_000
# NX = 3_000
# NY = 3_000

# if 32k x 32k single precision floats fit, then this amount of doubles fit
DX = int(sqrt(NX*NY/2))
print(DX)
STEPS = 100 # (32k)^2, T=100 => 16s
REPEATS = 5

UNROLL_LOOPS=True
USE_SINGLE_PRECISION=True
CONTIGUOUS = True
USE_MPI = False
MPI_PROCESSES = 1

def compile():
    subprocess.run(["cmake", "--build", "build", "--parallel"], check=True)

def measure_mlups(args=""):
    if USE_MPI:
        command = f"mpirun --bind-to none --map-by ppr:4:node -np {MPI_PROCESSES} ./main -nx {NX} -ny {NY} -of {0} -s {STEPS} --shear-wave-decay"+args
    else:
        command = f"./main -nx {NX} -ny {NY} -of {0} -s {STEPS} --shear-wave-decay"+args
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
    maxi = max(results)
    mini = min(results)
    return average, sample_standard_dev, maxi, mini

def overwrite_macros():
    with open("./src/macros.h","w") as f:
        f.write(f"""
#ifndef MACROS_H
#define MACROS_H

#define USE_SINGLE_PRECISION {"true" if USE_SINGLE_PRECISION else "false"}
#define UNROLL_LOOPS {"true" if UNROLL_LOOPS else "false"}
#define COALESCING {"true" if CONTIGUOUS else "false"}
#define USE_MPI {"true" if USE_MPI else "false"}

#endif""")

if __name__=="__main__":

    # normal run
    overwrite_macros()
    compile()
    pull_unrolled = measure_mlups()
    push_unrolled = measure_mlups(" -push")

    # run with non-contiguous reads/writes
    CONTIGUOUS = False
    overwrite_macros()
    compile()
    pull_uncont = measure_mlups()
    push_uncont = measure_mlups(" -push")
    CONTIGUOUS = True

    # run with non-unrolled loops
    UNROLL_LOOPS = False
    overwrite_macros()
    compile()
    pull_rolled = measure_mlups()
    push_rolled = measure_mlups(" -push")
    UNROLL_LOOPS = True

    # run with double precision
    USE_SINGLE_PRECISION = False
    NX, NY = DX, DX
    overwrite_macros()
    compile()
    pull_double = measure_mlups()
    push_double = measure_mlups(" -push")

    # reset macros
    UNROLL_LOOPS = True
    USE_SINGLE_PRECISION = True
    CONTIGUOUS = True
    overwrite_macros()

    # write out results
    name = datetime.now().strftime("%Y_%m_%d_%H:%M:%S")
    with open(f"benchmark{name}.results","w") as f:
        f.write(f"""
{NX}x{NY} for {STEPS} steps, {REPEATS} repeats, using mpi: {USE_MPI}

pull_unrolled   {pull_unrolled[0]} +- {pull_unrolled[1]}, min {pull_unrolled[2]}, max {pull_unrolled[3]}
push_unrolled   {push_unrolled[0]} +- {push_unrolled[1]}, min {push_unrolled[2]}, max {push_unrolled[3]}
pull_uncont     {pull_uncont[0]}   +- {pull_uncont[1]}, min {pull_uncont[2]}, max {pull_uncont[3]}
push_uncont     {push_uncont[0]}   +- {push_uncont[1]}, min {push_uncont[2]}, max {push_uncont[3]}
pull_rolled     {pull_rolled[0]}   +- {pull_rolled[1]}, min {pull_rolled[2]}, max {pull_rolled[3]}
push_rolled     {push_rolled[0]}   +- {push_rolled[1]}, min {push_rolled[2]}, max {push_rolled[3]}
pull_double     {pull_double[0]}   +- {pull_double[1]}, min {pull_double[2]}, max {pull_double[3]}
push_double     {push_double[0]}   +- {push_double[1]}, min {push_double[2]}, max {push_double[3]}
""")