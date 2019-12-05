import numpy as np
import matplotlib.pyplot as plt
from tikzplotlib import save as tikzsave

import os
cwd = os.getcwd()

from itertools import cycle
lines = ["-","--","-.",":"]
linecycler = cycle(lines)
# items = sorted(os.listdir(cwd))

samples = 300
gammas = [100]

fig, axes = plt.subplots(4, len(gammas), sharex=True, sharey=True)
# plt.figure()
for jj, it in enumerate(gammas):
    path = os.path.join(cwd, '30agents_review200_samplesize300_gamma{}'.format(it))
    if os.path.isdir(path):
        # Load the number of agents
        N = np.load(os.path.join(path, "agents.npy"))
        runs = np.load(os.path.join(path, "runs.npy"))

        # generate N colors
        colors = {}
        for idx, col in enumerate(plt.rcParams['axes.prop_cycle']):
            colors[idx] = col['color']

        np.random.seed(1)
        if idx < N:
            for i in range(idx, N):
                colors[i] = np.random.rand(3, 1).flatten()

        sequence = {}
        names = {}

        for i in range(N):
            filename = "agent_{}_sequence_run0.npy.npz".format(i)
            data = np.load(os.path.join(path, filename))
            aux = data['arr_1']
            name = data['arr_0']
            sequence[i] = np.zeros((runs, len(aux)))
            for run in range(runs):
                # Load the locally generated sequences
                filename = "agent_{}_sequence_run{}.npy.npz".format(i, run)
                fl = np.load(os.path.join(path, filename))
                sequence[i][run, :] = fl['arr_1']
                names[i] = str(fl['arr_0'])

        clrs = {'CNN': colors[0], 'HL2': colors[1], 'HL1': colors[2], 'SHL': colors[3]}
        CNNs=0
        HL2s=0
        HL1s=0
        SHLs=0
        for i in range(N):
            tests = len(sequence[i][0])
            x_c = np.linspace(0, 3, tests)
            if names[i] == 'CNN':
                CNNs+=1
                axes[0].plot(x_c, np.mean(sequence[i], axis=0), next(linecycler), color=clrs[names[i]], label="Agent {}".format(i))
            if names[i] == 'HL2':
                HL2s +=1
                axes[1].plot(x_c, np.mean(sequence[i], axis=0), next(linecycler), color=clrs[names[i]], label="Agent {}".format(i))
            if names[i] == 'HL1':
                HL1s+=1
                axes[2].plot(x_c, np.mean(sequence[i], axis=0), next(linecycler), color=clrs[names[i]], label="Agent {}".format(i))
            if names[i] == 'SHL':
                SHLs+=1
                axes[3].plot(x_c, np.mean(sequence[i], axis=0), next(linecycler), color=clrs[names[i]], label="Agent {}".format(i))
        for j in range(4):
            axes[j].set_ylim(0.5, 1)
            axes[j].grid()                

        # plt.xlabel(r"Epochs on $\mathcal{D}_s$")
        # plt.ylabel(r"Accuracy on $\mathcal{D}_{test}$")
        
            # if i == 0:
            #     axes[i, jj].set_title((r"$|\gamma|=$"+str(it)))
            # if jj == 0:
            #     axes[i, jj].set_ylabel(r"Accuracy on $\mathcal{D}_{test}$")
            
            # if i == 3:
            #     axes[i, jj].set_xlabel(r"Epochs on $\mathcal{D}_s$")
print(CNNs)
print(HL2s)
print(HL1s)
print(SHLs)
tikzsave("results_large.tex")          
plt.show()
