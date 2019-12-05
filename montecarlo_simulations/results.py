# Example of plot of the results
import numpy as np
import matplotlib.pyplot as plt
from tikzplotlib import save as tikzsave

import os
cwd = os.getcwd()

samples = 500
gammas = [0, 1, 10, 100, 1000]

fig, axes = plt.subplots(4, len(gammas), sharex=True, sharey=True)
for jj, it in enumerate(gammas):
    path = os.path.join(cwd, '4agents_review500_samplesize500_gamma{}'.format(it))
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
            for i in range(0, N):
                colors[i] = np.random.rand(3, 1).flatten()

        sequence = {}

        for i in range(N):
            filename = "agent_{}_sequence_run0.npy".format(i)
            aux = np.load(os.path.join(path, filename))
            sequence[i] = np.zeros((runs, len(aux)))
            for run in range(runs):
                # Load the locally generated sequences
                filename = "agent_{}_sequence_run{}.npy".format(i, run)
                sequence[i][run, :] = np.load(os.path.join(path, filename))

        for i in range(N):
            tests = len(sequence[i][0])
            x_c = np.linspace(0, 3, tests)
            axes[i, jj].plot(x_c, np.mean(sequence[i], axis=0), color=colors[i], label="Agent {}".format(i))
            axes[i, jj].fill_between(x_c,
                                     np.mean(sequence[i], axis=0) - 2*np.std(sequence[i], axis=0),
                                     np.mean(sequence[i], axis=0) + 2*np.std(sequence[i], axis=0),
                                     color=colors[i], alpha=0.4)
            axes[i, jj].set_ylim(0.5, 1)
            axes[i, jj].grid()
        
            if i == 0:
                axes[i, jj].set_title((r"$|\gamma|=$"+str(it)))
            if jj == 0:
                axes[i, jj].set_ylabel(r"Accuracy on $\mathcal{D}_{test}$")
            
            if i == 3:
                axes[i, jj].set_xlabel(r"Epochs on $\mathcal{D}_s$")

tikzsave("montecarlo_weights_samples500.tex")
plt.show()
