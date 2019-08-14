[![Codacy Badge](https://api.codacy.com/project/badge/Grade/96b3634f1abe48dc93b5ac19307bb394)](https://www.codacy.com/app/Neywiny/merge-sort?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=Neywiny/merge-sort&amp;utm_campaign=Badge_Grade)

# merge-sort
## Basic Usage 
```python main.py <iterations>```

outputs to resultsMerge```Distribution``` ```AUC```

Use DylAnalyzer.py to analyze the results across all iterations

[Link to documentation](https://neywiny.github.io/merge-sort/)

![alt text](https://github.com/Neywiny/merge-sort/blob/master/repository-pic.png)

## Reproducing a single Simulation
### merge
```python
from main import sort
resultss = sort((<dist>, <auc>, <n0>, <n1>))
```
Each element in resultss will be the results for that layer (such that in general index 0 is then there are groups of 2, index 1 is groups of 4, etc.)

The format for a result is:
```python
(auc, varEstimate, hanleyMcNeil, estimates, mseTrue, mseEmpiric, compLen, minSeps, pc) = resultss[layer index]
```

*   auc is the total accuracy 
*   varEstimate is the variance estimate
*   hanleyMcNeil is the current Hanley-McNeil variance estimate
*   estimates is the vector of Hanley-McNeil predictions from that layer onwards (so it will shrink in size as the layer number increases)
*   mseTruth is the MSE between the current ROC curve and the true ROC curve for the given distribution
*   mseEmpiric is the same as above just with that simulation's data set
*   compLen is th etotal number of comparisons
*   minSeps is the minimum number of comparisons between comparing the same image again for that image (it's a vector not a float)
*   pc is the percent of corrent comparisons from images of different distributions

To analyze the results, run ```DylAnalyzer.py <results filename>```

### elo

```python
# don't forget the ()
resultss = simulation_ELO_targetAUC((<dist>, <auc>, <n0>, <n1>), rounds=14)
```
Each element in resultss will be one round.

The format for a result is:
```python
(N, cnt, ncmp, var, auc, mseTruth, mseEmpiric, pc) = resultss[layer index]
```

*	N is n0
*	cnt is the number of comparisons done on images from different distributions
*	ncmp is th etotal number of comparisons
*	var is the success matrix variance estimate (it's bad)
*	auc is the total accuracy 
*	mseTruth is the MSE between the current ROC curve and the true ROC curve for the given distribution
*	mseEmpiric is the same as above just with that simulation's data set
*	pc is the percent of corrent comparisons from images of different distributions

## Doing a Reader Study

### Scale Rating System

Run ```DylScale.py <signal present directory> <signal absent directory> <n> <output file>```

For a quick analysis, you can run ```DylScale.py <input file>``` where the input file was the output file from the previous command

### AFC System

To do testing/training run ```DylAFC.py <target present directory> <target absent directory> <answers directory> <merge ip> <merger port> <n0> <n1> <log file>```

If you do not want to connect to a merge sort comparator, just give any value for ip and port

To do a merge sort study, run the same command with ip and port.

To start up the comparator, run ```DylComp.py <desired name of log file> <tcp port> <desired name of roc file>```

In the directory of DylComp a file called "figure.svg" will exist. If you open "dash.html" you will see a dashboard of how the reader is doing which is just automatically refreshing "figure.svg". It is recommended to keep "figure.svg" as a result. "dash.html" should not be seen by the reader while they are doing the study.

### Analysis

Results for reader study analysis are referenced with a json file. Each key should be a reader. Each reader should contain a list of 3 or 4 elements ordered as:

1.  The log from DylAFC
2.  The roc file from DylComp
3.  The log file from DylComp
4.  The log from DylScale (optional)

If there is no log file from DylScale, the analysis will not be able to show the results from the scale study.

To analyze the results, run ```DylAnalyzer <json file> <optional output file name>```

To run the full analysis on the ELO simulations, merge sort simulations, and reader studies, run ```FullAnalyzer.py```. It is not recommended to run this given that it relies on all three study types to be fully complete with no missing data, and as such is hardcoded to the result filenames used in the paper.