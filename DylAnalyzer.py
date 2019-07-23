import pickle
import numpy as np
np.seterr(divide='ignore', invalid='ignore')
import matplotlib.pyplot as plt
import matplotlib.lines as lines
from matplotlib.colors import LogNorm
from tqdm import trange, tqdm
from os import stat
from sys import getsizeof

def analyze(fileName, length, layers, justOne=False):
    avgAUC = np.zeros((layers,))
    avgSMVAR = np.zeros((layers,))
    avgnpVARs = np.zeros((layers,))
    avgComps = np.zeros((layers,))
    avgHanleyMNeil = np.zeros((layers,))
    avgMSETrues = np.zeros((layers,))
    avgMSEEmperic = np.zeros((layers,))
    avgErrorBars = np.zeros((layers, 6))
    avgEstimates = np.array([np.zeros((i)) for i in range(layers)])

    avgMinSeps = np.ones((layers, length))

    varEstimates = np.zeros((layers, 0))
    aucs = np.zeros((layers, 0))

    iters = 0
    fileLength = stat(fileName).st_size
    old = 0
    with open(fileName, "rb") as f, tqdm(total=fileLength, unit="B", unit_scale=True, disable=True) as pBar:
        unpickler = pickle.Unpickler(f)
        reshapeCount = 0
        while f.tell() < fileLength:
            iteration = unpickler.load()
            iters += 1
            iterEstimate = int(fileLength /(f.tell() / iters))
            if iters > len(aucs[0]):
                reshapeCount += 1
                new = np.zeros((layers, max(iterEstimate, iters + 1)))
                new[:aucs.shape[0], :aucs.shape[1]] = aucs
                aucs = new
                new = np.zeros((layers, max(iterEstimate, iters + 1)))
                new[:varEstimates.shape[0], :varEstimates.shape[1]] = varEstimates
                varEstimates = new
                del new
            for iLevel, (auc, varEstimate, hanleyMcNeil, lowBoot, highBoot, lowSine, highSine, smVAR, npVAR, *estimates, mseTrue, mseEmperic, compLen, minSeps) in enumerate(iteration):
                
                aucs[iLevel][iters - 1] = auc
                varEstimates[iLevel][iters - 1] = varEstimate
                print(npVAR, end=',')
                avgAUC[iLevel] += auc
                avgSMVAR[iLevel] += smVAR
                avgnpVARs[iLevel] += npVAR
                avgMSETrues[iLevel] += mseTrue
                avgMSEEmperic[iLevel] += mseEmperic
                avgComps[iLevel] += compLen
                avgHanleyMNeil[iLevel] += hanleyMcNeil
                avgErrorBars[iLevel] += [lowBoot, highBoot, lowSine, highSine, 0, 0]
                avgEstimates[layers - iLevel - 1] += estimates
                avgMinSeps[iLevel] += minSeps
            print()
            pBar.update(f.tell() - old)
            pBar.desc = f"{iters}/{iterEstimate}, {reshapeCount}, {getsizeof(unpickler)}"
            old = f.tell()
            if justOne:
                break
    avgMinSeps = avgMinSeps.transpose()[:,1:,]
    aucs = aucs[:,:iters]
    varEstimates = varEstimates[:,:iters]

    # axis=1 is across the simulations
    avgAUC = (avgAUC / iters).transpose()
    avgComps = avgComps // iters
    avgHanleyMNeil /= iters
    avgMSEEmperic /= iters
    avgEstimates /= iters
    avgMSETrues /= iters
    avgMinSeps /= iters
    avgnpVARs /= iters
    avgSMVAR /= iters
    avgErrorBars = np.transpose(avgErrorBars / iters)

    varEstimate = np.mean(varEstimates, axis=1)

    varAUCnp = np.var(aucs, ddof=1, axis=1)
    stdVarEstimate = np.sqrt(np.var(varEstimates, axis=1, ddof=1))

    remainder = int(bin(length)[3:], 2)
    thingies = [remainder, length / 2]
    for i in range(2, layers):
        thingies.append(thingies[-1] // 2)
    thingies = [1/(2*np.sqrt(thingy)) for thingy in thingies]

    avgErrorBars[4] = [np.sin(np.arcsin(np.sqrt(auc)) - thingies[i])**2 for i, auc in enumerate(avgAUC)]
    avgErrorBars[5] = [np.sin(np.arcsin(np.sqrt(auc)) + thingies[i])**2 for i, auc in enumerate(avgAUC)]

    return aucs, varEstimate, avgAUC, avgSMVAR, avgnpVARs, avgMSETrues, avgMSEEmperic, avgComps, avgHanleyMNeil, avgErrorBars, avgEstimates, avgMinSeps, varAUCnp, stdVarEstimate, iters

if __name__ == "__main__":
    
    length = 256
    layers = 8

    aucs, varEstimate, avgAUC, avgSMVAR, avgnpVARs, avgMSETrues, avgMSEEmperic, avgComps, avgHanleyMNeil, avgErrorBars, avgEstimates, avgMinSeps, varAUCnp, stdVarEstimate, iters = analyze("resultsMergeExponential65", length, layers)

    labels = [f'{np.median(list(filter(lambda x: x != 0, avgMinSeps[0]))):3.02f}']
    for val in np.median(avgMinSeps, axis=0)[1:]:
        labels.append(f'{val:3.02f}')

    slopeFirst = (varEstimate[1]/avgHanleyMNeil[1]) - (varEstimate[0]/avgHanleyMNeil[0])
    slopeTotal = slopeFirst / 3

    hanleyMcNeilToVarEstimate = [avgHanleyMNeil[i] * (1 + i * slopeTotal) for i in range(layers)]

    #print(varEstimate)
    #print(varAUCnp)
    #print(avgAUC)
    #print(avgComps)
    #exit()
    tickLabels = ['0'] + [str(int(n)) for n in avgComps]
    #print(avgAUC, avgVAR, avgComps)
    xVals = avgComps
    #xVals = [*range(1, len(avgAUC) + 1)]
    fig = plt.figure()
    ax1 = fig.add_subplot(2, 3, 1)
    #ax1 = fig.add_subplot(1, 1, 1)
    ax1.plot(xVals, avgAUC, 'b', lw=0, label='Confidence Interval:')
    """for iter in trange(1000):
        for level in range(len(aucs[0])):
            ax1.scatter(level, aucs[iter][level])"""
    #ax1.errorbar(xVals, avgAUC, yerr=[avgAUC - avgErrorBars[4], avgErrorBars[5] - avgAUC], capsize=10, c='g', label="arcsine(avg x)")
    #ax1.errorbar(xVals, avgAUC, yerr=[avgAUC - avgErrorBars[2], avgErrorBars[3] - avgAUC], capsize=10, c='r', label="avg arcsine(x)")
    ax1.errorbar(xVals, avgAUC, yerr=np.sqrt(varEstimate), capsize=5, c='r', lw=0, elinewidth=2, label="$\pm\sqrt{var_{estimate}}$")
    ax1.errorbar(xVals, avgAUC, yerr=[avgAUC - avgErrorBars[0], avgErrorBars[1] - avgAUC], capsize=10, c='b', label="bootstrap")
    #ax1.set_xticklabels(tickLabels)
    ax1.set_ylim(top=0.96)
    ax1.legend()

    ax1.set_ylabel('AUC', color='b')
    #ax1.ticklabel_format(useOffset=False)
    ax1.set_title("Average AUC per Layer")
    #plt.show()
    #xVals = [*range(0, len(avgComps) + 1)]
    xVals = avgComps
    ax2 = fig.add_subplot(2, 3, 2)
    ax2.plot(xVals, avgSMVAR, 'b.', ls=':', label='VAR sm')
    ax2.plot(xVals, varAUCnp, 'g.', ls='--', lw=5, label='$var_{real}$')
    ax2.plot(xVals, avgnpVARs, '.', c='orange', ls='--', label='VAR np')
    ax2.errorbar(xVals, varEstimate, yerr=stdVarEstimate, c='r', marker='.', ls=':', lw=2, label='$var_{estimate}$')
    #ax2.plot(xVals, avgHanleyMNeil, 'c.', ls='-', lw=2, label='HmN Variance')
    #ax2.set_xticklabels(tickLabels)
    #ax2.plot(xVals[1:], hanleyMcNeilToVarEstimate, 'm.', ls=':', lw=2, label='HmN estimate')
    for layer in range(1, layers):
        estimate = avgEstimates[layer - 1]
        for i, point in enumerate(estimate):
            pass
            #ax2.text(layer + 1, point, str(i), fontsize=12, horizontalalignment='center', verticalalignment='center')
    ax2.legend()
    ax2.set_title("Variance Estimate per Layer")

    ax3 = fig.add_subplot(2, 3, 3)
    info = [-1 for i in range(layers - 1)]
    for layer in range(layers - 1):
        try:
            info[layer] = ((1/varEstimate[layer + 1]) - (1/varEstimate[layer]))/(avgComps[layer + 1] - avgComps[layer])
        except ZeroDivisionError:
            print(varEstimate, avgComps)
    ax3.plot(xVals[1:], info, marker='.')
    #ax3.set_xticklabels(tickLabels)
    ax3.set_title("Information Gained per Comparison per Layer")
    #ax3.set_yscale('log')

    ax4 = fig.add_subplot(2, 2, 3)
    ax4.plot([0, len(avgComps)], [0, max(avgComps)], 'b:')
    ax4.plot(list(range(9)), [0, *avgComps], 'r.-', label='comparisons')
    ax4.set_ylabel('Comparisons', color='r') 
    ax4.set_yticks([0] + [int(avgComp) for avgComp in avgComps])
    ax4.set_title("Average Comparisons per Layer")

    ax5 = fig.add_subplot(2, 2, 4)
    plot = ax5.imshow(avgMinSeps,norm=LogNorm(), extent=[0, length, 0, length], aspect=0.5)
    ax5.set_xticks([*range(length//(2*layers), length, int((layers / (layers - 1))*(length//layers)))])

    start, end = ax5.get_xlim()
    step = length / (layers - 1)
    #ax5.set_xticks(list(range(step, length, step)))
    ax5.set_xticks(np.arange(start + (step / 2), end + (step / 2), step))
    ax5.set_xticklabels(tickLabels[1:])

    cbaxes = fig.add_axes([0.91, 0.13, 0.01, 0.31])
    cbar = fig.colorbar(plot, cax=cbaxes)

    ax5.set_title("Average Distance Between Compairisons per ID per Layer")

    plt.subplots_adjust(wspace=0.45)
    #plt.show()
