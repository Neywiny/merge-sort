import pickle
import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange, tqdm

iters = 19*2560

layers = 8

for withComp in [True, False]:
    avgAUC = [0 for i in range(layers)]
    avgVARsm = [0 for i in range(layers)]
    avgComps = [0 for i in range(layers)]


    avgMinSeps = [[0 for level in range(layers) ] for i in range(256)]

    VARnp = [[0 for __ in range(layers)] for _ in range(iters)]
    aucs = [[0 for __ in range(layers)] for _ in range(iters)]
    #i = ''
    #if True:
    for i in trange(1, 20) if withComp else trange(20, 39):
        with open("results/results"+str(i), "rb") as f:
            results = pickle.load(f)
            for iIter, iteration in enumerate(tqdm(results)):
                for iLevel, (auc,smVAR,npVAR,compLen,minSeps) in enumerate(iteration[:-1]):
                    aucs[iIter][iLevel] = auc
                    avgAUC[iLevel] += auc
                    avgVARsm[iLevel] += smVAR
                    VARnp[iIter][iLevel] = npVAR
                    avgComps[iLevel] += compLen
                    for (key, val) in minSeps:
                        if val != 257:
                            if avgMinSeps[key][iLevel] != 257:
                                avgMinSeps[key][iLevel] += val / iters
                            else:
                                avgMinSeps[key][iLevel] = val / iters
                        else:
                            avgMinSeps[key][iLevel] = 257
            del results #pleeeeeeeeeeeease get out of memory
    avgAUC = list(map(lambda x: x/iters, avgAUC[:-1]))
    varAUCnp = np.var(aucs, ddof=1, axis=0)
    avgVARsm = list(map(lambda x: x/iters, avgVARsm[:-1]))
    avgComps = list(map(lambda x: x/iters, avgComps[:-1]))
    # first layer are all NoneType's by definition
    avgMinSeps = np.array(avgMinSeps[1:])

    #print(avgAUC, avgVAR, avgComps)
    xVals = [*range(1, len(avgComps) + 1)]
    fig = plt.figure()
    ax1 = fig.add_subplot(2, 4, 1)
    ax1.plot(xVals, avgAUC, 'g.-', label='AUC')
    ax1.set_ylabel('AUC', color='g')
    ax1.ticklabel_format(useOffset=False)

    ax21 = fig.add_subplot(2, 4, 2)
    ax21.plot(avgVARsm, 'b.-', label='VAR sm')
    ax22 = fig.add_subplot(2, 4, 3)
    ax22.plot(np.mean(VARnp, axis=0), 'r.-', label='VARnp of AUC')
    ax23 = fig.add_subplot(2, 4, 4)
    ax23.plot(varAUCnp[:-1], 'm.-', label='VAR np')
    ax21.set_ylabel('VAR', color='b')
    ax22.set_ylabel('VAR numpy', color='r')
    ax23.set_ylabel('VAR numpy of AUCs', color='m')
    #ax2.set_yscale('log')

    ax3 = fig.add_subplot(2, 2, 3)
    ax3.plot([1, len(avgComps) ], [min(avgComps), max(avgComps)], 'b')
    ax3.plot(xVals, avgComps, 'r.-', label='comparisons')
    ax3.set_ylabel('Comparisons', color='r') 
    
    ax4 = fig.add_subplot(2, 2, 4)
    plot = ax4.imshow(np.log10(avgMinSeps), extent=[0, 256, 0, 256], aspect=0.5)
    ax4.set_xticks([*range(0, 256, 256//8)])
    ax4.set_xticklabels([0, 1, 2, 3, 4, 5, 6, 7, 8])
    cbaxes = fig.add_axes([0.91, 0.12, 0.01, 0.33])
    fig.colorbar(plot, cax=cbaxes)
    fig.suptitle(f'Withcomp: {withComp}')
    plt.subplots_adjust(wspace=0.45)
plt.show()
