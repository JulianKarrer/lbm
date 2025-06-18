#!/usr/bin/env python3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.optimize import curve_fit

import subprocess
from io import StringIO

plt.rcParams.update({"text.usetex": True,})

# NX = 1024
# NY = 1024
# TOTAL_STEPS = 200_000
# DUMP_EVERY = int(TOTAL_STEPS/(60*20))
NX = 128
NY = 128
TOTAL_STEPS = 1000
DUMP_EVERY = 1
OMEGA = 1.4
U_0 = 0.1
DATA_FILE_NAME = "lid-driven"+str(NX)+"x"+str(NY)+"s"+str(TOTAL_STEPS)+".npy"

# compile the program
subprocess.run(["cmake", "--build", "build"], check=True)

# run the program, get the output
res = subprocess.run([
        "./main", 
        "-w", str(OMEGA),
        "-nx", str(NX),
        "-ny", str(NY),
        "-of", str(DUMP_EVERY),
        "-s", str(TOTAL_STEPS+1),
        "-ov", # output velocity magnitudes
        # "-ld", # lid driven cavity
        "-u", str(U_0),
        "-rho", str(1.0),
    ], cwd="build",stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,check=False).stdout

# parse the program output into floats
frames = []
for step in res.split("#"):
    try:
        df = pd.read_csv(StringIO(step), header=None, dtype=np.float64)
        frames.append(df.values)
    except:
        ...
print("number of frames:", len(frames))
frames = np.array(frames)
# np.save(DATA_FILE_NAME, frames)
# frames = np.load(DATA_FILE_NAME)
print(frames[10])

# setup animation
x = np.arange(NX)
y = np.arange(NY)
X, Y = np.meshgrid(x, y)

# settings
TITLE = str(NX)+"x"+str(NY)+' LB-2DQ9-PBC Shear Wave Decay ('+str(DUMP_EVERY*len(frames))+" steps)"
CMAP = "Spectral_r"
LEVELS = 200
CBAR_TITLE = r"$\left|\vec{\mathbf{u}}\right|$"

# Initial contour; levels are fixed then
fig, ax = plt.subplots()
fig.set_size_inches(12.8, 10.8, forward=True)
levels = np.linspace(0, U_0, LEVELS)
contour = ax.contourf(X, Y, frames[0], levels=levels, cmap=CMAP)
cbar = fig.colorbar(contour, ax=ax)
cbar.set_label(CBAR_TITLE)
ax.set_title(TITLE)
ax.set_xlabel("x")
ax.set_ylabel("y")

def update(frame_idx):
    ax.clear()
    cf = ax.contourf(X, Y, frames[frame_idx], levels=levels, cmap=CMAP)
    ax.set_title(TITLE)
    return cf.collections

anim = animation.FuncAnimation(
    fig,                    # the figure to update
    update,                 # update function
    frames=frames.shape[0], # number of frames
    interval=16,            # delay between frames in ms
    blit=True               # blitting for performance
)


plt.show()

# try:
#     # REQUIRES FFMPEG
#     anim.save('lid-driven-test.mp4', writer='ffmpeg', dpi=100, fps=60)
# except:
#     ...

# print("DONE")