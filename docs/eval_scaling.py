from math import sqrt
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({
    "text.usetex": True,
})

from matplotlib.ticker import MaxNLocator
ax = plt.figure().gca()
ax.xaxis.set_major_locator(MaxNLocator(integer=True))


UPDATES = [
    102412800400,
    102412800400*2,
    102412800400*4,
    102412800400*8,
]

CORES=[
    1,
    2,
    4,
    8,
]

TIMES = [
    [  # 102412800400
        [4938737713],
        [4946679562],
        [4946838428],
        [4947433708],
        [4944974620],
    ],
    [ # 102412800400
        [4947333379, 4972078122],
        [4934143218, 4964172517],
        [4936463915, 4959944110],
        [4935941628, 4957703392],
        [4946999521, 4959541016],
    ],
    [ # 102412800400*4
        [4980964207,5000335086,5017284300,5030172608],
        [4955313333,4970482691,5033195609,5023434002],
        [4939310324,4999479757,5008172495,5020724028],
        [4949603451,4965577201,5021582076,5026432585],
        [4994256410,4983995321,4998276334,5025536895],
    ],
    [ # 102412800400*8
        [5010958006,5010149481,5025617316,5055190613,5068984809,5072055686,5087561597,5079828355,],
        [4989439009,5024069857,5046808786,5032683631,5048647235,5065839770,5093845968,5082944457,],
        [5000941679,5015857763,5018362102,5037520155,5078299106,5062737571,5064529740,5084788330,],
        [4996386805,5020018384,5028669545,5049770886,5035725449,5050344464,5068214484,5094432249,],
        [4993052680,5028858083,5037914352,5056909851,5057860037,5062013585,5089525306,5077169178,],
    ]
]

mlups = [
    [
        1e3*UPDATES[test] / np.mean(TIMES[test][i])
        for i in range(len(TIMES[test]))
    ]
    for test in range(len(TIMES))
]

n = len(mlups)
avgs = [np.mean(ml) for ml in mlups]
mins = [np.min(ml) for ml in mlups]
maxs = [np.max(ml) for ml in mlups]
stddev = [np.std(ml, ddof=1) for ml in mlups] # Bessel corrected
stderr = [stddev/np.sqrt(len(ml)) for ml,stddev in zip(mlups,stddev)] # σ / sqrt(N)
weak_scaling = [float(avgs[i]/(avgs[0]*CORES[i])) for i in range(n)]

# https://en.wikipedia.org/wiki/Propagation_of_uncertainty 
# weak_scaling = f = A/B case for ratio in weak scaling, with A=avgs[i], B=avgs[b]
weak_scaling_errors = [
    float(weak_scaling[i] * np.sqrt(
        (stderr[i]/avgs[i])**2 + (stderr[0]/avgs[0])**2
    ))
    for i in range(n)
]

plt.title("Weak Scaling")
plt.errorbar(CORES, weak_scaling, weak_scaling_errors, fmt='o-')
plt.xlabel(r"Number of Processes $N$")
plt.ylabel(r"Weak Scaling ratio $\frac{T_N}{N \cdot T_1}$")
plt.show()


print("AVGS", avgs)
print("MINS", mins)
print("MAXS", maxs)
print("DEVS", stddev)
print("WEAK SCALING", weak_scaling)
print("WEAK SCALING ERRORS", weak_scaling_errors)
