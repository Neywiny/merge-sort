import numpy as np
from numpy import matlib as mb
from ROC1 import *
from scipy.special import erfinv
from scipy.stats import norm
from math import sqrt
from warnings import filterwarnings
from DylMath import MSE
from tqdm import tqdm, trange
from pickle import dumps, Unpickler
from struct import unpack
import sys
argv = sys.argv
filterwarnings('ignore')

def AUCVAR(x1, x0):
    sm = successmatrix(x1, np.transpose(x0))
    return np.mean(sm), unbiasedMeanMatrixVar(sm)

def queueReader(queue, total):
    with tqdm(total=total, smoothing=0) as pbar, open(f"resultsElo{argv[1].title()}{argv[2][2:]}", "wb") as f:
        while True:
            msg = queue.get()
            if msg == 'DONE':
                break
            else:
                pbar.update()
                pbar.set_description(str(len(msg)))
                f.write(msg)

def simulation_ELO_targetAUC(retStats=False, queue=None):
    ##
    # Function for the simulation of ELO rating given an AUC of 0.8 (most of it, hard-coded), 
    # the input to the function is N (the number of samples on the rating study).
    #
    #
    # @Author: Francesc Massanes (fmassane@iit.edu)
    # @Version: 0.1 (really beta)

    if isinstance(retStats, (tuple, list)):
        rounds = retStats[2]
        queue = retStats[1]
        retStats = retStats[0]

    N = 256 # use this for matching to merge sort
    N //= 2 # this function doubles N so here we halve it
    
    ## 
    # DATA GENERATION 
    #
    
    K1 = 400
    K2 = 32

    
    #mu1 = 1
    #mu0 = 0
    #si0 = 0.84
    #si1 = sqrt( 2*((mu1-mu0)**2/K - si0**2 /2 ) )
    
    
    #neg = np.random.normal(mu0, si0, (N, 1))
    #plus = np.random.normal(mu1, si1, (N, 1))
    #neg = np.random.normal(0, 1, (N, 1))
    #plus = np.random.normal(1.7, 1, (N, 1))
    with open("/dev/urandom", "rb") as f:
        seed = unpack("<I", f.read(4))[0]
    np.random.seed(seed=seed)

    if len(sys.argv) > 1:
        AUC = float(sys.argv[2])
        if sys.argv[3] == 'exponential':
            neg = np.random.exponential(size=(N, 1))
            sep = abs(AUC/(1-AUC))
            plus = np.random.exponential(scale=sep, size=(N, 1))
        elif sys.argv[3] == 'normal':
            sep = norm.ppf(AUC)*(2**0.5)
            neg = np.random.normal(0, 1, (N, 1))
            plus = np.random.normal(sep, 1, (N, 1))
        else:
            print("invalid argument", sys.argv[3])
            while True: pass
    else:
        neg = np.random.normal(0, 1, (N, 1))
        plus = np.random.normal(1.7, 1, (N, 1))

    x0 = np.array(neg)[:,0]
    x1 = np.array(plus)[:,0]
    empiricROC = rocxy(x1, x0)
    scores = np.append(neg, plus)
    truth = np.append(mb.zeros((N, 1)), mb.ones((N, 1)), axis=0)
    
    AUC_orig, _ = AUCVAR(neg, plus)
    #print(f'AUC original: {AUC_orig:.3f}\n')
    
    #
    ## 
    # PRE-STABLISHED COMPARISONS
    #
    if not 'rounds' in locals():
        rounds = 14
    M = rounds*N
    rating = np.append(mb.zeros((N, 1)), mb.zeros((N, 1)), axis=0)
    
    cnt = 0
    ncmp = 0
    results = list()
    for round in range(1, rounds+1):
        toCompare = mb.zeros((2*N, 1))
    
        if round == 1:
            # option A: only compare + vs -
            arr = list(range(N))
            #np.random.shuffle(arr)
            toCompare[0::2] = np.array(arr, ndmin=2).transpose()
            arr = list(range(N, 2 * N))
            #np.random.shuffle(arr)
            toCompare[1::2] = np.array(arr, ndmin=2).transpose()
        else:
            # option B: everything is valid
            arr = list(range(2 * N))
            np.random.shuffle(arr)
            toCompare = np.array(arr, ndmin=2).transpose()
    
        for i in range(1, 2*N, 2):
            a = int(toCompare[i - 1])
            b = int(toCompare[i])
    
            QA = 10**(int(rating[a]) / K1)
            QB = 10**(int(rating[b]) / K1)
    
            EA = QA / (QA+QB)
            EB = QB / (QA+QB)
    
            if scores[a] < scores[b]:
                SA = 0
                SB = 1
            else:
                SA = 1
                SB = 0

            if ( SA == 1 and truth[a] == 1 ):
                cnt = cnt + 1
            if ( SB == 1 and truth[b] == 1 ):
                cnt = cnt +1
            ncmp = ncmp+1
    
            rating[a] = rating[a] + K2 * ( SA - EA )
            rating[b] = rating[b] + K2 * ( SB - EB )
    
        x0 = np.array(rating[0:N])[:,0]
        x1 = np.array(rating[N:])[:,0]
        auc, var = AUCVAR(x1, x0)

        roc = rocxy(x1, x0)
        mseTruth, mseEmperic, auc = MSE(sep, sys.argv[3], roc, empiricROC)
        if retStats == True:
            pass
            #yield roc, mseTruth, mseEmperic, empiricROC
        if queue != None:
            queue.put(dumps((N, cnt, ncmp, var, auc, mseTruth, mseEmperic)))
        else:
            results.append((N, cnt, ncmp, var, auc, mseTruth, mseEmperic))
    if queue == None:
        return results
if __name__ == '__main__':
    if len(argv) > 1:
        test = 1
    else:
        test = 2
    if test == 1:
        #simulation_ELO_targetAUC(200)
        from multiprocessing import Manager, Process, Pool
        import os
        from time import sleep
        #from p_tqdm import p_umap
        """iters = int(argv[3]) if len(argv) > 1 else 8
        rounds = 14
        manager = Manager()
        with Pool() as p:
            queue = manager.Queue()
            readerProcess = Process(target=queueReader, args=((queue, iters*rounds)))
            readerProcess.daemon = True
            readerProcess.start()
            p.map(simulation_ELO_targetAUC, ((i, queue, rounds) for i in range(iters)))
        queue.put('DONE')
        readerProcess.join()"""
        iters = int(sys.argv[1])
        for dist in ['normal', 'exponential']:
            for AUC in [0.65, 0.85, 0.95]:
                if dist == 'normal':
                    sep = norm.ppf(AUC)*(2**0.5)
                elif dist == 'exponential':
                    sep = abs(AUC/(1-AUC))
                results = list()
                if len(sys.argv) > 1:
                    iters = int(sys.argv[1])
                    if len(sys.argv) < 3:
                        sys.argv.append(AUC)
                        sys.argv.append(dist)
                    else:
                        sys.argv[-2] = AUC
                        sys.argv[-1] = dist
                    ids = [*range(iters)]
                    topBar = tqdm(total=iters, smoothing=0, bar_format="{percentage:3.0f}% {n_fmt}/{total_fmt} {remaining}, {rate_fmt}")
                    botBar = tqdm(total=iters, smoothing=0, bar_format="{bar}")
                    with Pool() as p:
                        for eloResults in p.imap_unordered(simulation_ELO_targetAUC, ids):
                            topBar.update()
                            botBar.update()
                            for result in eloResults:
                                results.append(dumps(result))
                    botBar.close()
                    topBar.close()
                    print('\n')
                else:
                    retMid = False
                    iters = 1
                    results = [pickle.dumps(sort(0, i)) for i in range(iters)]
                #change output file if requested to do so
                print("waiting for lock")
                locked = False
                while not locked:
                    try:
                        lock = open(".lock", "x")
                        print("made lock")
                        locked = True
                    except FileExistsError as e:
                        sleep(0.1)
                try:
                    with open(f'resultsElo{dist.title()}{int(AUC*100)}','ab') as f:
                        print("have lock")
                        f.writelines(results)
                except BaseException as e:
                    print(e)
                finally:
                    lock.close()
                    os.remove(".lock")



    elif test == 2:
        animation = False
        if animation:
            import matplotlib.pyplot as plt
            from matplotlib.animation import FuncAnimation
            from matplotlib.animation import PillowWriter
            results = simulation_ELO_targetAUC(True)
            fig, ax = plt.subplots()
            fig.set_tight_layout(True)
            pbar = tqdm(total=len(results))
            def update(i):
                pbar.update()
                label = f"timestep {i}"
                ax.clear()
                roc = rocxy(*results[i])
                ax.plot(roc['x'], roc['y'])
                ax.set_title(f"{i:02d}")
                return label, ax
            anim = FuncAnimation(fig, update, frames=np.arange(0, 100), interval=100)
            anim.save("rocs.gif", writer=PillowWriter(fps=10))
            pbar.close()
        else:
            import matplotlib.pyplot as plt
            from apng import APNG
            from DylSort import treeMergeSort, genD0D1
            from DylComp import Comparator
            from DylData import continuousScale
            from DylMath import genROC, avROC
            seed = 15
            data, D0, D1 = continuousScale(128, 128)
            comp = Comparator(data, level=0, rand=True, seed=seed)
            comp.genRand(len(D0), len(D1), 7.72, 'exponential')

            np.random.seed(seed)
            im = APNG()
            fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)
            fig.suptitle("Pass number, MSE true, MSE empirical")
            x = np.linspace(0, 1, num=200)
            y = x**(1/7.72)
            ax1.set_aspect('equal', 'box')
            ax2.set_aspect('equal', 'box')
            elo = simulation_ELO_targetAUC(True)
            merge = treeMergeSort(data, comp, d0d1 = (D0, D1), combGroups=False)
            plt.tight_layout()
            for i in trange(8):
                roc, mseTheo, mseEmp, empiricROC = next(elo)
                ax1.plot(x, y, linestyle='--', label='true', lw=3)
                ax1.plot(empiricROC['x'], empiricROC['y'], linestyle=':', lw=2, label='empirical')
                ax1.plot(roc['x'], roc['y'], label='predicted')
                ax1.legend(loc=4)
                ax1.set_title(f"ELO\n{i+1}, {mseTheo[0]*1000:02.3f}E(-3), {mseEmp[0]*1000:02.3f}E(-3)")
                
                groups = next(merge)
                rocs = []
                for group in groups:
                    gD0, gD1 = genD0D1((D0, D1), group)
                    rocs.append(genROC(group, gD0, gD1))
                roc = avROC(rocs)
                mseTheo, mseEmp, auc = MSE(7.72, zip(*roc)), MSE(7.72, zip(*roc), zip(empiricROC['x'], empiricROC['y']))
                ax2.plot(x, y, linestyle='--', label='true', lw=3)
                ax2.plot(empiricROC['x'], empiricROC['y'], linestyle=':', lw=2, label='empirical')
                ax2.plot(*roc, label='predicted')
                ax2.legend()
                ax2.set_title(f"merge\n{i+1}, {mseTheo[0]*1000:02.3f}E(-3), {mseEmp[0]*1000:02.3f}E(-3)")
                
                plt.savefig(f"both")
                im.append_file(f"both.png", delay=1000)
                ax1.clear()
                ax2.clear()
            im.save("both.png")

