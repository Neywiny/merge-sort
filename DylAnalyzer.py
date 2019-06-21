import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from tqdm import trange, tqdm

cores = 19
passes = 2560
iters = cores*passes

length = 256

layers = 6

avgAUC = [0 for i in range(layers)]
avgVARsm = [0 for i in range(layers)]
avgComps = [0 for i in range(layers)]


avgMinSeps = [[0 for level in range(layers) ] for i in range(length)]

VARnp = [[-1 for __ in range(layers)] for _ in range(iters)]
aucs = [[0 for __ in range(layers)] for _ in range(iters)]
#i = ''
#if True:
for i in trange(1, cores + 1):
    with open("results/results"+str(i), "rb") as f:
        results = pickle.load(f)
        for iIter, iteration in enumerate(results):
            for iLevel, (auc,smVAR,npVAR,compLen,minSeps) in enumerate(iteration[:-1]):
                aucs[iIter + ((i-1)*passes)][iLevel] = auc
                avgAUC[iLevel] += auc
                avgVARsm[iLevel] += smVAR
                VARnp[iIter + ((i-1)*passes)][iLevel] = npVAR
                avgComps[iLevel] += compLen
                for (key, val) in minSeps:
                    if val != (2 * length):
                        if avgMinSeps[key][iLevel] != (2 * length):
                            avgMinSeps[key][iLevel] += val / iters
                        else:
                            avgMinSeps[key][iLevel] = val / iters
                    else:
                        avgMinSeps[key][iLevel] = (2 * length)
        del results #pleeeeeeeeeeeease get out of memory
avgAUC = list(map(lambda x: x/iters, avgAUC))
varAUCnp = np.var(aucs, ddof=1, axis=0)
avgVARsm = list(map(lambda x: x/iters, avgVARsm))
avgComps = list(map(lambda x: x/iters, avgComps))
VARnp = np.mean(VARnp, axis=0)

varEstimate = [VARnp[0]]
varEstimate.extend([-1 for i in range(layers - 1)])

#print(layers, len(varEstimate), len(avgVARsm), len(VARnp))

for layer in range(1, layers - 1):
    varEstimate[layer] = (avgVARsm[layer] + VARnp[layer]) / 2
varEstimate[-1] = avgVARsm[-1]

#print(avgAUC, avgVAR, avgComps)
xVals = [*range(1, len(avgAUC) + 1)]
fig = plt.figure()
ax1 = fig.add_subplot(2, 3, 1)
#ax1.plot(xVals, avgAUC, 'g.-', label='AUC')
ax1.errorbar(xVals, avgAUC, yerr=np.sqrt(varEstimate))
ax1.set_ylabel('AUC', color='b')
ax1.ticklabel_format(useOffset=False)
ax1.set_title("Average AUC per layer")

xVals = [*range(0, len(avgComps) + 1)]

ax2 = fig.add_subplot(2, 3, 2)
#ax2.plot(avgVARsm, 'b.', ls=':', label='VAR sm')
#ax2.plot(VARnp, 'r.', ls='-.', label='VARnp of AUC')
#ax2.plot(varAUCnp[:-1], 'm.', ls='--', label='VAR np')
ax2.plot(xVals[1:], varEstimate, 'g.', ls='-', label='variance estimate')
ax2.legend()
ax2.set_title("Variance Estimate per layer")

ax3 = fig.add_subplot(2, 3, 3)
info = [-1 for i in range(layers - 1)]
for layer in range(layers - 1):
    try:
        info[layer] = ((1/varEstimate[layer + 1]) - (1/varEstimate[layer]))/(avgComps[layer + 1] - avgComps[layer])
    except ZeroDivisionError:
        print(varEstimate, avgComps)
ax3.plot(xVals[2:], info)
ax3.set_title("Information Gained per Comparison per Layer")

ax4 = fig.add_subplot(2, 2, 3)
ax4.plot([0, len(avgComps)], [0, max(avgComps)], 'b:')
ax4.plot(xVals, [0, *avgComps], 'r.-', label='comparisons')
ax4.set_ylabel('Comparisons', color='r') 
ax4.set_title("Average Comparisons per Layer")

ax5 = fig.add_subplot(2, 2, 4)
plot = ax5.imshow(avgMinSeps,norm=LogNorm(), extent=[0, length, 0, length], aspect=0.5)
ax5.set_xticks([*range(0, length + length//layers, length//layers)])
ax5.set_xticklabels([*range(layers + 1)])
cbaxes = fig.add_axes([0.91, 0.13, 0.01, 0.31])
cbar = fig.colorbar(plot, cax=cbaxes)

ax5.set_title("Average Distance Between Compairisons per ID per Layer")

plt.subplots_adjust(wspace=0.45)
plt.show()
