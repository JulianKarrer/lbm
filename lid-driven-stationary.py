#!/usr/bin/env python3

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm

from math import ceil, log10
from concurrent.futures import ProcessPoolExecutor
import os
import subprocess
from io import StringIO

plt.rcParams.update({"text.usetex": True,})

RECOMPILE = True
RUN = True

NX = 2**10 # 2048
NY = 2**10 # 2048
TOTAL_STEPS = 500_000#10_000_000
# DUMP_EVERY = 1 #1_000_000-1
DUMP_EVERY = int(TOTAL_STEPS/(60*20))
OMEGA = 1.7
U_LID = 0.1
RHO = 1.0
DATA_FILE_NAME = "lid-driven"+str(NX)+"x"+str(NY)+"s"+str(TOTAL_STEPS)+".csv"
OUT_DIR = "./out2"

LINE_BY_LINE = False

if RECOMPILE:
    # recompile the program
    subprocess.run(["cmake", "--build", "build"], check=True)

if RUN:
    # run the program, get the output
    # ./build/main -w 1.7 -nx 64 -ny 64 -of 99999 -s 100000 -ld -u 0.1 -rho 1 -f asdf 
    subprocess.run([
            "./main", 
            "-w", str(OMEGA),
            "-nx", str(NX),
            "-ny", str(NY),
            "-of", str(DUMP_EVERY),
            "-s", str(TOTAL_STEPS),
            "-ovf", # output velocity field
            "-ld", # lid driven cavity scene
            # "--shear-wave-decay",
            "-u", str(U_LID),
            "-rho", str(RHO),
            "-f", DATA_FILE_NAME,
        ], 
        cwd="build",
        check=False,
    )


def plot_quiver(u: np.ndarray, v: np.ndarray, i:int, stride = max(NX // 32, 1)):
    NX, NY = u.shape
    x = np.arange(NY)
    y = np.arange(NX)
    X, Y = np.meshgrid(x, y)
    Xs = X[::stride, ::stride]
    Ys = Y[::stride, ::stride]
    Us = v[::stride, ::stride]
    Vs = u[::stride, ::stride]
    arrow_length_in_axes = 100
    scale = U_LID / arrow_length_in_axes
    fig, ax = plt.subplots(figsize=(8, 6))

    _quiver = ax.quiver(
        Xs, Ys, Us, Vs,
        scale=scale,
        scale_units='xy',
        width=0.0025,
        color='tab:blue'
    )

    ax.set_title(r"Lid-Driven Cavity $"+ str(NX) +r"\times "+ str(NY) +r"$ Velocity Field at $t=" + str(DUMP_EVERY*i).zfill(ceil(log10(TOTAL_STEPS))) + r"$")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_aspect("equal")

    return fig

def plot_streamlines(u: np.ndarray, v: np.ndarray, i:int, stride: int = 1):
    NX, NY = u.shape
    x = np.arange(NY)
    y = np.arange(NX)
    X, Y = np.meshgrid(x, y)
    fig, ax = plt.subplots(figsize=(8, 6))
    # fix seed points
    n_seeds_x = NY // 8  # adjust for density
    n_seeds_y = NX // 8
    seed_x = np.linspace(0, NY - 1, n_seeds_x)
    seed_y = np.linspace(0, NX - 1, n_seeds_y)
    seed_points = np.array(np.meshgrid(seed_x, seed_y)).reshape(2, -1).T

    _strm = ax.streamplot(
        X, Y, v, u,
        density=1.0,
        linewidth=1.0,
        integration_direction='forward',
        start_points=seed_points
    )
    ax.set_title(r"Lid-Driven Cavity $"+ str(NX) +r"\times "+ str(NY) +r"$ Velocity Field at $t=" + str(DUMP_EVERY*i).zfill(ceil(log10(TOTAL_STEPS))) + r"$")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_aspect("equal")
    return fig


def plot_magnitude(u: np.ndarray, v: np.ndarray, i:int):
    NX, NY = u.shape
    x = np.arange(NY)
    y = np.arange(NX)
    X, Y = np.meshgrid(x, y)
    speed = np.sqrt(u**2 + v**2)
    fig, ax = plt.subplots(figsize=(8, 6))
    # define levels between 0 and ulid
    levels = np.linspace(0., U_LID, 256)
    norm = BoundaryNorm(levels, ncolors=256, clip=True)
    pcm = ax.pcolormesh(
        X, Y, speed,
        cmap="Spectral_r",
        shading='gouraud',
        norm=norm
    )
    cbar = fig.colorbar(pcm, ax=ax)
    cbar.set_label(r"Velocity magnitude $\left|\vec{\mathbf{u}}\right|$")
    
    ax.set_title(r"Lid-Driven Cavity $"+ str(NX) +r"\times "+ str(NY) +r"$ Velocity Magintude at $t=" + str(DUMP_EVERY*i).zfill(ceil(log10(TOTAL_STEPS))) + r"$")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_aspect("equal")
    return fig

def run(u,v,i):
    print(i)
    fig_m = plot_magnitude(u,v,i)
    fig_m.savefig(OUT_DIR+"/mag-"+DATA_FILE_NAME[:-3]+str(i).zfill(6)+".png")
    plt.close(fig_m)

    fig_q = plot_quiver(u,v,i)
    fig_q.savefig(OUT_DIR+"/quiver-"+DATA_FILE_NAME[:-3]+str(i).zfill(6)+".png")
    plt.close(fig_q)

    fig_s = plot_streamlines(u,v,i)
    fig_s.savefig(OUT_DIR+"/stream-"+DATA_FILE_NAME[:-3]+str(i).zfill(6)+".png")
    plt.close(fig_s)

# parse the program output 
frames = []
buf = ""
second = False
i = 0
if not os.path.exists(OUT_DIR):
    os.makedirs(OUT_DIR)
with ProcessPoolExecutor() as executor, open(f"./build/{DATA_FILE_NAME}", "r") as file:
    futures = []
    for line in file:
        if "#" in line:
            df = pd.read_csv(StringIO(buf), header=None, dtype=np.float64)
            frames.append(df.values)
            buf = ""
            if second:
                uv = np.array(frames)
                # plot in an off-thread and keep processing
                futures.append(executor.submit(run, uv[-1], uv[-2], i))
                # run(uv[-1], uv[-2], i,)
                # reset buffer to keep memory consumption low
                frames = []
                second = False
                i += 1
            else:
                second = True
        else:
            buf += line
    # wait for futures to finish
    for future in futures:
            future.result()
