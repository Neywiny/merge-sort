from random import random
from math import tanh

from DylUtils import *

def nearlySorted(maxi: int, factor: int) -> list:
    
    """creates a list of nearly sorted data ranging from 0-maximum
    the smoothness of the data is decided by 'factor'"""

    l: list = [i for i in range(maxi)]
    output: list = []
    pos: int = 0
    # basic scheme: move through the array taking steps back randomly
    for i in range(maxi):
        if random() > (0.5 + (factor/100)):
            pos += 1
        else:
            pos -= 1
        if pos >= len(l):
            pos = len(l) - 1
        if pos < 0:
            pos = 0
        output.append(l[pos])
        l.pop(pos)
    return output
def randomDisease(lMax: int) -> list:
    data: list = list(range(lMax))
    output: tuple = [data.pop(0) if (random() >  1+tanh(1.5*(i/lMax - 1))) else data.pop(len(data) - 1) for i in range(lMax)]
    return output

if __name__ == "__main__":
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt
    import numpy as np
    outputs: list = []
    maxi: int = 100

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for factor in range(50):
        output: list = nearlySorted(maxi, factor)
        outputs.append(output)
        
    y = np.arange(50)
    x = np.arange(maxi)
    X, Y = np.meshgrid(x, y)
    Z = np.array(outputs)

    ax.plot_surface(X, Y, Z)

    ax.set_xlabel('i')
    ax.set_ylabel('factor')
    ax.set_zlabel('l[pos]')
    plt.show()