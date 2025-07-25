#!/usr/bin/env python3

import numpy as np
from math import sin, pi
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import subprocess

plt.rcParams.update({"text.usetex": True,})
plt.rcParams["figure.figsize"] = (4*2.5,3*2.5)

# compile the program
subprocess.run(["cmake", "--build", "build", "--parallel"], check=True)

DUMP_EVERY = 500
NX = 1000
NY = 1000
TOTAL_STEPS = 200_000
OMEGA_STEPS = 5
OMEGAS = [(1./OMEGA_STEPS)*i for i in range(1,OMEGA_STEPS+1)]
MPI_RANKS = 1


# Set up plots
TITLE_FONT_SIZE = 14
fig_max, ax_max = plt.subplots()
ax_max.set_xlabel(r"$t$")
ax_max.set_ylabel(r"$\left|\vec{\mathbf{v}}\right|$")
ax_max.set_title(
    r"Time evolution of the velocity amplitude for differing $\omega$",
    fontsize=TITLE_FONT_SIZE
)

fig_om_nu, ax_om_nu = plt.subplots()
ax_om_nu.set_xlabel(r"$\omega$")
ax_om_nu.set_ylabel(r"$\nu$")
ax_om_nu.set_title(
    r"Measured kinematic viscosity $\nu$ as a function of relaxation parameter $\omega$", 
    fontsize=TITLE_FONT_SIZE
)
ax_om_nu.set_xscale("log")
ax_om_nu.set_yscale("log")

fig_om_nu_err, ax_om_nu_err = plt.subplots()
ax_om_nu_err.set_xlabel(r"$\omega$")
ax_om_nu_err.set_ylabel(r"$\left|\nu - \nu^*\right|$")
ax_om_nu_err.set_title(
    r"Absolute error of measured kinematic viscosity",
    fontsize=TITLE_FONT_SIZE
)
ax_om_nu_err.set_yscale("log")


measured_nus = []
nu_from_omega = lambda omega:(1./3.)*(1./omega - 0.5)
decays = []

for omega in OMEGAS:
    # run the program, get the output
    command = f"{f"mpirun -n {MPI_RANKS} " if MPI_RANKS>1 else ""}./main -w {omega} -nx {NX} -ny {NY} -of {DUMP_EVERY} -s {TOTAL_STEPS} -omv"
    print("command running:", command)
    output = subprocess.run(command.split(" "), cwd="build",stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,check=False)
    res = output.stdout
    print(output.stderr)

    # collect the output velocity amplitudes
    max_us = []
    for step in res.split(",")[:-1]:
        try:
            val = float(step)
            max_us += [val]
        except:
            ...

    ts = [DUMP_EVERY*i for i in range(len(max_us))]

    # fit an exponential model to the velocity amplitude over time
    exp = lambda x,a,nu,c: a * np.exp(-nu * (2*np.pi/NY)**2 * x)+c
    try:
        (a0_fit, measured_nu, c_fit), pcov = curve_fit(exp, ts, max_us, (0.2**2,1,0))
    except:
        measured_nu = 0
    expected_nu = nu_from_omega(omega)
    # print("measured A_0=",a0_fit," measured nu=", measured_nu, "expected nu=",expected_nu, "nu abs error=", abs(expected_nu-measured_nu))
    measured_nus += [measured_nu]

    # plot the expected and measured velocity amplitude evolution
    ax_max.plot(ts, max_us, label=r"$\omega = "+'{:3.2f}'.format(omega)+r"$")
    decays += [{"ts":ts, "max_us":max_us, "omega":omega}]

# plot the measured viscosity as a funciton of omega and the errors
ax_om_nu.plot(OMEGAS, [nu_from_omega(om) for om in OMEGAS], "-r", label="expected viscosity")
ax_om_nu.scatter(OMEGAS, measured_nus, label="measured viscosity")
ax_om_nu_err.plot(OMEGAS, [abs(nu_from_omega(om)-nu) for (om,nu) in zip(OMEGAS, measured_nus)])

print("expected\n",[nu_from_omega(om) for om in OMEGAS])
print("measured\n",measured_nus)
print("decays",decays)

# save the plots to disk
ax_max.legend()
ax_om_nu.legend()
plt.show()
# fig_max.savefig("shear-umax-evolution.jpg", dpi=300)
# fig_om_nu.savefig("shear-nu-over-omega.jpg", dpi=300)
# fig_om_nu_err.savefig("shear-nu-error.jpg", dpi=300)

