from math import sqrt
import numpy as np
import matplotlib.pyplot as plt

UPDATES = [
    102412800400,
    102412800400*2,
    102412800400*4
]

CORES=[
    1,
    2,
    4,
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
weak_scaling = [avgs[i]/(avgs[0]*CORES[i]) for i in range(n)]

# https://en.wikipedia.org/wiki/Propagation_of_uncertainty 
# weak_scaling = f = A/B case for ratio in weak scaling, with A=avgs[i], B=avgs[b]
weak_scaling_errors = [
    weak_scaling[i] * np.sqrt(
        (stderr[i]/avgs[i])**2 + (stderr[0]/avgs[0])**2
    )
    for i in range(n)
]

plt.title("Weak Scaling")
plt.errorbar(CORES, weak_scaling, weak_scaling_errors, fmt='o-')
plt.show()


print("AVGS", avgs)
print("MINS", mins)
print("MAXS", maxs)
print("DEVS", stddev)
print("WEAK SCALING", weak_scaling)
