Merge Sort
==========

|forthebadge made-with-python|

|Codacy Badge| |Python 3.6,3.7| |Maintenance| |Open Source Love png2|

.. figure:: https://github.com/Neywiny/merge-sort/blob/master/repository-pic.png
   :alt: alt text

.. contents::

Documentation
-------------

`Link to documentation <https://merge-sort.readthedocs.io/>`__

Reproducing many simulations
----------------------------

merge/elo
~~~~~~~~~

``python3 main.py/elo.py <iters> <distributions> <aucs>``

Where distributions and aucs are each delimited by commas and no spaces.

This will output a single results file per distribution per auc, ex.
resultsMergeNormal85, resultsEloExponential95. This command is also safe
to be run accross many different nodes accessing the same file system,
and has been tested with up to 19 nodes running simulations.

Reproducing a single Simulation
-------------------------------

merge
~~~~~

.. code:: python

    from main import sort
    resultss = sort((<dist>, <auc>, <n0>, <n1>))

Each element in resultss will be the results for that layer (such that
in general index 0 is then there are groups of 2, index 1 is groups of
4, etc.)

The format for a result is:

.. code:: python

    (auc, varEstimate, hanleyMcNeil, estimates, mseTrue, mseEmpiric, compLen, minSeps, pc) = resultss[layer index]

where

-  auc is the total accuracy
-  varEstimate is the variance estimate
-  hanleyMcNeil is the current Hanley-McNeil variance estimate
-  estimates is the vector of Hanley-McNeil predictions from that layer
   onwards (so it will shrink in size as the layer number increases)
-  mseTruth is the MSE between the current ROC curve and the true ROC
   curve for the given distribution
-  mseEmpiric is the same as above just with that simulation's data set
-  compLen is th etotal number of comparisons
-  minSeps is the minimum number of comparisons between comparing the
   same image again for that image (it's a vector not a float)
-  pc is the percent of corrent comparisons from images of different
   distributions

To analyze the results, run
``python3 DylAnalyzer.py <results filename>``

elo
~~~

.. code:: python

    # don't forget the ()
    resultss = simulation_ELO_targetAUC((<dist>, <auc>, <n0>, <n1>), rounds=14)

Each element in resultss will be one round.

The format for a result is:

.. code:: python

    (N, cnt, ncmp, var, auc, mseTruth, mseEmpiric, pc) = resultss[layer index]

where

-  N is n0 (basically just for record keeping)
-  cnt is the number of comparisons done on images from different
   distributions
-  ncmp is th etotal number of comparisons
-  var is the success matrix variance estimate (it's bad)
-  auc is the total accuracy
-  mseTruth is the MSE between the current ROC curve and the true ROC
   curve for the given distribution
-  mseEmpiric is the same as above just with that simulation's data set
-  pc is the percent of corrent comparisons from images of different
   distributions

Doing a Reader Study
--------------------

Scale Rating System
~~~~~~~~~~~~~~~~~~~

Run
``python3 DylScale.py <signal present directory> <signal absent directory> <n> <output file> <offset (defualts to 0)>``.
This will output the results to the output filename with the start time
in Unix time and ".csv" after. This is because the sale ratings are all
independant from each other so if you want to do half at one time and
half at a later time you can, just change the offset parameter and
append the new file to the old one.

For a quick analysis, you can run ``python3 DylScale.py <input file>``
where the input file was the output file from the previous command.

AFC System
~~~~~~~~~~

To do testing/training run
``python3 DylAFC.py <target present directory> <target absent directory> <answers directory> <merge ip> <merger port> <n0> <n1> <log file>``

If you do not want to connect to a merge sort comparator, just give any
value for ip and port

To do a merge sort study, run the same command with ip and port.

To start up the comparator, run
``python3 DylComp.py <desired name of log file> <tcp port> <desired name of roc file>``

In the directory of DylComp a file called "figure.svg" will exist. If
you open "dash.html" you will see a dashboard of how the reader is doing
which is just automatically refreshing "figure.svg". It is recommended
to keep "figure.svg" as a result. "dash.html" should not be seen by the
reader while they are doing the study.

Analysis
~~~~~~~~

Results for reader study analysis are referenced with a json file. Each
key should be a reader. Each reader should contain a list of 3 or 4
elements ordered as:

1. The log from DylAFC
2. The roc file from DylComp
3. The log file from DylComp
4. The log from DylScale (optional)

Example:

.. code:: json

    {
        "Reader A":[
            "resA/log.csv",
            "resA/rocs",
            "resA/compA.csv",
            "resA/scaleA123456.123.csv"
        ],
        "Reader B":[
            "resB/log.csv",
            "resB/rocs",
            "resB/compB.csv",
            "resB/scaleB456789.012.csv"
        ],
        "Reader C":[
            "resC/log.csv",
            "resC/rocs",
            "resC/compC.csv",
            "resC/scaleC345678.901.csv"
        ]
    }

If there is no log file from DylScale, the analysis will not be able to
show the results from the scale study.

To analyze the results, run
``python3 DylAnalyzer.py <json file> <optional output file name>``

Graphs and Where to Find Them
-----------------------------

-  Graph of the green/red success matrix ROC curve ->
   ``python3 DylSort.py 1 <n0> <n1> <directory to save file into (optional)>``
-  Dashboard of a merge sort simulation file ->
   ``python3 DylAnalyzer.py 1 <filename>``
-  Reader study p vals and time analysis ->
   ``python3 DylAnalyzer.py 2 <results json filename> <names.txt filename (in case it was moved or renamed; required)> <graph output filename (optional)>``
-  Canonical bottom up merge sort vs tree based merge sort ->
   ``python3 DylSort.py 5``
-  Average ROC for each layer as a merge simulation progresses ->
   ``python3 DylSort.py 3 <overlapping (defualt True)>``
-  ROC curves for merge sort vs elo -> ``python3 elo.py``

.. |forthebadge made-with-python| image:: http://ForTheBadge.com/images/badges/made-with-python.svg
   :target: https://www.python.org/
.. |Codacy Badge| image:: https://api.codacy.com/project/badge/Grade/96b3634f1abe48dc93b5ac19307bb394
   :target: https://www.codacy.com/app/Neywiny/merge-sort?utm_source=github.com&utm_medium=referral&utm_content=Neywiny/merge-sort&utm_campaign=Badge_Grade
.. |Python 3.6,3.7| image:: https://img.shields.io/badge/python-3.6%20%7C%203.7-blue
   :target: https://www.python.org/downloads/release/python-370/
.. |Maintenance| image:: https://img.shields.io/badge/Maintained%3F-yes-green.svg
   :target: https://GitHub.com/Naereen/StrapDown.js/graphs/commit-activity
.. |Open Source Love png2| image:: https://badges.frapsoft.com/os/v2/open-source.png?v=103
   :target: https://github.com/ellerbrock/open-source-badges/
