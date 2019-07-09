import pickle
import numpy as np
np.seterr(divide='ignore', invalid='ignore')
import matplotlib.pyplot as plt
import matplotlib.lines as lines
from matplotlib.colors import LogNorm
from tqdm import trange, tqdm
from os import stat

length = 130
remainder = int(bin(length)[3:], 2)

layers = 8

avgAUC = np.zeros((layers, 1))
avgComps = np.zeros((layers, 1))
avgEstimate = np.array([np.zeros((i + 1)) for i in range(layers - 1)])

avgMinSeps = np.zeros((length, layers - 1))

hanleyMcNeils = [list() for __ in range(layers)]
varEstimates = [list() for __ in range(layers)]
aucs = [list() for __ in range(layers)]
errorBars = [list() for __ in range(layers)]

iters = 0
fileLength = stat("results").st_size

with open("results", "rb") as f:
    with tqdm() as pBar:
        unpickler = pickle.Unpickler(f)
        while f.tell() < fileLength:
            iteration = unpickler.load()
            iters += 1
            for iLevel, (auc, varEstimate, hanleyMcNeil, lowBoot, highBoot, lowSine, highSine, *estimates, compLen, minSeps) in enumerate(iteration):

                aucs[iLevel].append(auc)
                hanleyMcNeils[iLevel].append(hanleyMcNeil)
                varEstimates[iLevel].append(varEstimate)
                errorBars[iLevel].append([lowBoot, highBoot, lowSine, highSine, 0, 0])

                for layer, estimate in enumerate(estimates, start=layers-len(estimates) - 1):
                    avgEstimate[layer][iLevel] += estimate

                avgAUC[iLevel] += auc
                avgComps[iLevel] += compLen

                for key, val in enumerate(minSeps):
                    if val != (2 * length):
                        avgMinSeps[key][iLevel - 1] += val
            pBar.update(1)

# axis=1 is across the simulations
avgAUC = (avgAUC / iters).transpose()[0]
avgComps = avgComps / iters
avgMinSeps /= iters
avgEstimate /= iters

avgErrorBars = np.mean(errorBars, axis=1).transpose()
avgHanleyMcNeil = np.mean(hanleyMcNeils, axis=1)
varEstimate = np.mean(varEstimates, axis=1)

varAUCnp = np.var(aucs, ddof=1, axis=1)
stdVarEstimate = np.sqrt(np.var(varEstimates, axis=1))


thingies = [remainder, length]
for i in range(2, layers):
    thingies.append(thingies[-1] // 2)
thingies = [1/(2*np.sqrt(thingy)) for thingy in thingies]
avgErrorBars[4] = [np.sin(np.arcsin(np.sqrt(auc)) - thingies[i])**2 for i, auc in enumerate(avgAUC)]
avgErrorBars[5] = [np.sin(np.arcsin(np.sqrt(auc)) + thingies[i])**2 for i, auc in enumerate(avgAUC)]

labels = [f'{np.median(list(filter(lambda x: x != 0, avgMinSeps[0]))):3.02f}']
for val in np.median(avgMinSeps, axis=0)[1:]:
    labels.append(f'{val:3.02f}')

slopeFirst = (varEstimate[1]/avgHanleyMcNeil[1]) - (varEstimate[0]/avgHanleyMcNeil[0])
slopeTotal = slopeFirst / 3

hanleyMcNeilToVarEstimate = [avgHanleyMcNeil[i] * (1 + i * slopeTotal) for i in range(layers)]

#print(varEstimate)
#print(varAUCnp)
#print(avgAUC)
#print(avgComps)
#exit()

#print(avgAUC, avgVAR, avgComps)
xVals = [*range(1, len(avgAUC) + 1)]
fig = plt.figure()
ax1 = fig.add_subplot(2, 3, 1)
#ax1 = fig.add_subplot(1, 1, 1)
ax1.plot(xVals, avgAUC, 'b.-', label='AUC')
"""for iter in trange(1000):
    for level in range(len(aucs[0])):
        ax1.scatter(level, aucs[iter][level])"""
ax1.errorbar(xVals, avgAUC, yerr=[avgAUC - avgErrorBars[4], avgErrorBars[5] - avgAUC], capsize=10, c='g')
ax1.errorbar(xVals, avgAUC, yerr=[avgAUC - avgErrorBars[2], avgErrorBars[3] - avgAUC], capsize=10, c='r')
ax1.errorbar(xVals, avgAUC, yerr=[avgAUC - avgErrorBars[0], avgErrorBars[1] - avgAUC], capsize=10, c='b')

ax1.set_ylabel('AUC', color='b')
ax1.ticklabel_format(useOffset=False)
ax1.set_title("Average AUC per layer")
#plt.show()
xVals = [*range(0, len(avgComps) + 1)]

ax2 = fig.add_subplot(2, 3, 2)
#ax2.plot(avgVARsm, 'b.', ls=':', label='VAR sm')
#ax2.plot(varAUCnp, 'r.', ls='-.', label='VARnp of AUC')
#ax2.plot(VARsNP[:-1], 'm.', ls='--', label='VAR np')
ax2.errorbar(xVals[1:], varEstimate, yerr=stdVarEstimate, c='r', marker='.', ls='-', lw=2, label='variance estimate')
ax2.plot(xVals[1:], avgHanleyMcNeil, 'c.', ls=':', lw=2, label='HmN Variance')
ax2.plot(xVals[1:], hanleyMcNeilToVarEstimate, 'm.', ls=':', lw=2, label='HmN estimate')
for layer in range(1, layers):
    estimate = avgEstimate[layer - 1]
    for i, point in enumerate(estimate):
        ax2.text(layer + 1, point, str(i), fontsize=12, horizontalalignment='center', verticalalignment='center')
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
#ax3.set_yscale('log')

ax4 = fig.add_subplot(2, 2, 3)
ax4.plot([0, len(avgComps)], [0, max(avgComps)], 'b:')
ax4.plot(xVals, [0, *avgComps], 'r.-', label='comparisons')
ax4.set_ylabel('Comparisons', color='r') 
ax4.set_title("Average Comparisons per Layer")

ax5 = fig.add_subplot(2, 2, 4)
plot = ax5.imshow(avgMinSeps,norm=LogNorm(), extent=[0, length, 0, length], aspect=0.5)
ax5.set_xticks([*range(length//(2*layers), length, int((layers / (layers - 1))*(length//layers)))])
ax5.set_xticklabels([*map(lambda i: str(i + 2) + '\n' +  labels[i], range(layers - 1))])
cbaxes = fig.add_axes([0.91, 0.13, 0.01, 0.31])
cbar = fig.colorbar(plot, cax=cbaxes)

ax5.set_title("Average Distance Between Compairisons per ID per Layer")

plt.subplots_adjust(wspace=0.45)
plt.show()
