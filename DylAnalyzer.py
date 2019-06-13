import pickle
import numpy as np
import matplotlib.pyplot as plt

results = []
#i = ''
#if True:
for i in range(1, 20):
    with open("results/results"+str(i), "rb") as f:
        results.extend(pickle.load(f))

print(len(results))

avgAUC = [0 for i in range(len(results[0]))]
avgVARsm = [0 for i in range(len(results[0]))]
avgComps = [0 for i in range(len(results[0]))]


avgMinSeps = [[0 for i in range(len(results[0][0][4]))] for level in range(len(results[0]) - 1)]

VARnp = [[0 for __ in range(len(results[0]))] for _ in range(len(results))]
aucs = [[0 for __ in range(len(results[0]))] for _ in range(len(results))]

for iIter, iteration in enumerate(results):
    for iLevel, (auc,smVAR,npVAR,compLen,minSeps) in enumerate(iteration[:-1]):
        aucs[iIter][iLevel] = auc
        avgAUC[iLevel] += auc
        avgVARsm[iLevel] += smVAR
        VARnp[iIter][iLevel] = npVAR
        avgComps[iLevel] += compLen
        for (key, val) in minSeps:
            if val != 257:
                if avgMinSeps[iLevel][key] != 257:
                    avgMinSeps[iLevel][key] += val / len(results)
                else:
                    avgMinSeps[iLevel][key] = val / len(results)
            else:
                avgMinSeps[iLevel][key] = 257


avgAUC = list(map(lambda x: x/len(results), avgAUC[:-1]))
varAUCnp = np.var(aucs, ddof=1, axis=0)
avgVARsm = list(map(lambda x: x/len(results), avgVARsm[:-1]))
avgComps = list(map(lambda x: x/len(results), avgComps[:-1]))
# first layer are all NoneType's by definition
avgMinSeps = np.array(avgMinSeps[1:])

#print(avgAUC, avgVAR, avgComps)
xVals = [*range(1, len(avgComps) + 1)]
fig = plt.figure()
ax1 = fig.add_subplot(2, 2, 1)
ax1.plot(xVals, avgAUC, 'g.-', label='AUC')
ax1.set_ylabel('AUC', color='g')

ax2 = fig.add_subplot(2, 2, 2)
ax2.plot(avgVARsm, 'b.-', label='VAR sm')
ax2twin = ax2.twinx()
ax2twin.plot(varAUCnp[:-1], 'g.-', label='VAR np')
ax2twin.plot(np.mean(VARnp, axis=0), 'r.-', label='VAR np (avg of iters)')
ax2.legend(loc=7)
ax2twin.legend(loc=3)
ax2.set_ylabel('VAR', color='b')
ax2.set_yscale('log')

ax3 = fig.add_subplot(2, 2, 3)
ax3.plot([1, len(avgComps) ], [min(avgComps), max(avgComps)], 'b')
ax3.plot(xVals, avgComps, 'r.-', label='comparisons')
ax3.set_ylabel('Comparisons', color='r') 

ax4 = fig.add_subplot(2, 2, 4)
plot = ax4.imshow(avgMinSeps, extent=[0, 256, 0, 256], aspect=0.5)
ax4.set_yticks([*range(0, 256, 256//7)])
ax4.set_yticklabels(reversed([0, 1, 2, 3, 4, 5, 6, 7]))
cbaxes = fig.add_axes([0.91, 0.12, 0.01, 0.33])
plt.colorbar(plot, cax=cbaxes)

plt.show()