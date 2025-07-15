#!/usr/bin/env python3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.optimize import curve_fit # type: ignore

import subprocess
from io import StringIO

plt.rcParams.update({"text.usetex": True,})

NX = 512
NY = 512
TOTAL_STEPS = 100_000
DUMP_EVERY = int(TOTAL_STEPS/(60*10))
OMEGA = 1.4
U_0 = 0.1
DATA_FILE_NAME = "shear-wave"+str(NX)+"x"+str(NY)+"s"+str(TOTAL_STEPS)

# compile the program
subprocess.run(["cmake", "--build", "build"], check=True)

# run the program, get the output
command = f"./main -w {OMEGA} -nx {NX} -ny {NY} -of {DUMP_EVERY} -s {TOTAL_STEPS+1} -u {U_0} -rho 1.0 -ov"
print("command running:", command)
output = subprocess.run(command.split(" "),cwd="build",stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,check=False)
print(output.stderr)
res = output.stdout

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
print(frames.shape)

# np.save(DATA_FILE_NAME+".npy", frames)
# frames = np.load(DATA_FILE_NAME+".npy")

# setup animation
x = np.arange(NX)
y = np.arange(NY)
X, Y = np.meshgrid(x, y)

# settings
TITLE = str(NX)+"x"+str(NY)+' LB-2DQ9-PBC Shear Wave Decay ('+str(DUMP_EVERY*len(frames))+" steps)"
CMAP = "Spectral_r"
LEVELS = 100
CBAR_TITLE = r"$\left|\vec{\mathbf{u}}\right|$"

# Initial contour; levels are fixed then
fig, ax = plt.subplots()
fig.set_size_inches(12.8, 10.8, forward=True)
levels = np.linspace(0, U_0*1.05, LEVELS)
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
    return cf.collections # type: ignore

anim = animation.FuncAnimation(
    fig,                    # the figure to update
    update,                 # update function
    frames=frames.shape[0], # number of frames
    interval=16,            # delay between frames in ms
    blit=True               # blitting for performance
)


plt.show()

try:
    # REQUIRES FFMPEG
    anim.save(DATA_FILE_NAME+".mp4", writer='ffmpeg', dpi=100, fps=60)
except:
    ...

print("DONE")