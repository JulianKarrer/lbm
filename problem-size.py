
from math import sqrt
import subprocess
from datetime import datetime

N_MAX = 32_000**2

DIVIDES = 4

STEPS = 100
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
    # write macros and compile
    overwrite_macros()
    compile()

    # measure lups for successively smaller grid sizes
    res = "extent,total_nodes,mlups\n"
    sizes = []
    mlups_all = []
    for i in range(1,DIVIDES+1):
        NX = int(sqrt(N_MAX / i))
        NY = NX
        mlups = measure_mlups()
        res += f"{NX},{NX*NY},{mlups}\n"
        sizes += [NX*NY]
        mlups_all += [mlups]
    
    print(res)
    print("sizes\n",sizes)
    print("mlups\n",mlups_all)

    # # write out results
    # name = datetime.now().strftime("%Y_%m_%d_%H:%M:%S")
    # with open(f"problemsize{name}.results","w") as f:
    #     f.write(res)