Merge Sort
==========

|forthebadge made-with-python|

|Python 3.6,3.7| |rtd-badge| |Maintenance|

|Codacy Badge| |Open Source Love png2| |licence|

.. contents::

.. figure:: https://github.com/Neywiny/merge-sort/blob/master/repository-pic.png
   :alt: alt text

What is this repository?
------------------------
This repository is the code behind the paper "Efficiently calculating ROC curves, their areas, and uncertainty from forced-choice studies with finite samples".

Primarily, this code is used for running reader studies on provided images. 
Code is provided for the standard scale ranking system, where a score is given by the reader on each image in sequence. The results for this study are dumped to a CSV-esque file of the image's path, the score given, and the time it took for the reader to evaluate the image.

This code also does the study described in the paper, a 2AFC study with the software taking care of which images to display at what times. There are a few result files for this. A results.csv is a table where each row (except the first which is a header) contains the two images as their numbers given by the software displayed and then the image chosen. log2.csv is more similar to the scale result file, where it gives the images as their paths, the image chosen, and the amount of time taken.
Along with that, at every layer the statistics of the layer are outputted.

Terminology
-----------
* n0: the number of images without a signal/disease
* n1: the number of images with a signal/disease
* merge: often used as a shorthand for the mergesort algorithm or a mergesort simulation/study
* elo: the Massanes and Brankov method

How to use the repository
-------------------------
The first step is to download it. This can be done with the git program (if you want to make changes to this repository) or as a zip (if you just want the code). Either way, click the "Code" button near the top of the page (not top of this document). It should be a solid color.

For help, click the question mark that appears after clicking the button. If the zip file is desired, click "Download ZIP". Then follow the installation section's instructions.

After reading this section, decide what you want to do. If you want to run a reader study, proceed to the section on that. Likewise for simulations, proceed to the respective sections.

When running files, on Windows "python" should be used, on other systems "python3" should be used before any of the .py files.

All filed are run from the command line, though there are some user interfaces such as for the studies or analyses without output files.

Installation
------------

Clone/download the repository. To install the requirements run

``pip3 install -r requirements.txt``.

This will install all the requirements needed to run everything in the paper and in this readme.

API Reference
-------------

`Link to API reference <https://merge-sort.readthedocs.io/>`__


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
``python3 DylAFC.py <target present directory> <target absent directory> <answers directory> <comparator ip> <comparator port> <n0> <n1> <log file>``

Where the answers directory is the directory of the target present images with the target highlighted for training the reader. These need to be in the same alphabetical order as the target present directory images.

If you do not want to connect to a merge sort comparator, just give any
value for ip and port. This is used when you only need AFC training or training on what signals look like.

Note that once the "study" button is pressed the program will try to connect to the study, so do not press it unless you are doing a study. It will just wait forever.

To do a merge sort study, run the same command with ip and port.

To start up the comparator, run
``python3 DylComp.py <desired name of log file> <tcp port> <desired name of roc file>``

In the directory of DylComp a file called "figure.svg" will exist. If
you open "dash.html" you will see a dashboard of how the reader is doing
which is just automatically refreshing "figure.svg". It is recommended
to keep "figure.svg" as a results file. "dash.html" and "figure.svg" should not be seen by the
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

``python3 DylAnalyzer.py 2 <json file> <names.txt> [optional output file name]``

Where names.txt is the path of all the images in the study. They must match up with the paths in the scale.ccsv results file.

Reproducing many simulations
----------------------------

merge/elo
~~~~~~~~~

``python3 <main.py or elo.py> <iters> <distributions> <aucs>``

Where distributions and aucs are each delimited by commas and no spaces.

This will output a single results file per distribution per auc, ex.
resultsMergeNormal85 or resultsEloExponential95. This command is also safe
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
``python3 DylAnalyzer.py 1 <results filename> <total number of images> <layers>``

ELO
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

Graphs and Where to Find Them
-----------------------------

-  Graph of the green/red success matrix ROC curve ->
   ``python3 DylSort.py 1 <n0> <n1> <directory to save file into (optional)>``
-  Dashboard of a merge sort simulation file ->
   ``python3 DylAnalyzer.py 1 <filename> <total number of images> <layers>``
-  Reader study p vals and time analysis ->
   ``python3 DylAnalyzer.py 2 <results json filename> <names.txt filename (in case it was moved or renamed; required)> <graph output filename (optional)>``
-  Canonical bottom up merge sort vs tree based merge sort ->
   ``python3 DylSort.py 5``
-  Average ROC for each layer as a merge simulation progresses ->
   ``python3 DylSort.py 3 <overlapping (defualt True)>``
-  ROC curves for merge sort vs ELO -> ``python3 elo.py``

Disclaimer
----------

This software and documentation (the "Software") were developed at the Food and Drug Administration (FDA) by employees of the Federal Government in the course of their official duties. Pursuant to Title 17, Section 105 of the United States Code, this work is not subject to copyright protection and is in the public domain. Permission is hereby granted, free of charge, to any person obtaining a copy of the Software, to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, or sell copies of the Software or derivatives, and to permit persons to whom the Software is furnished to do so. FDA assumes no responsibility whatsoever for use by other
parties of the Software, its source code, documentation or compiled executables, and makes no guarantees, expressed or implied, about its quality, reliability, or any other characteristic. Further, use of this code in no way implies endorsement by the FDA or confers any advantage in regulatory decisions. Although this software can be redistributed and/or modified freely, we ask that any derivative works bear some notice that they are derived from it, and any modified versions bear some notice that they have been modified.


.. |forthebadge made-with-python| image:: http://ForTheBadge.com/images/badges/made-with-python.svg
   :target: https://www.python.org/
.. |Codacy Badge| image:: https://api.codacy.com/project/badge/Grade/96b3634f1abe48dc93b5ac19307bb394
   :target: https://www.codacy.com/app/Neywiny/merge-sort?utm_source=github.com&utm_medium=referral&utm_content=Neywiny/merge-sort&utm_campaign=Badge_Grade
.. |Python 3.6,3.7| image:: https://img.shields.io/badge/python-3.6%20%7C%203.7-blue?style=for-the-badge&logo=python&logoColor=yellow
   :target: https://www.python.org/downloads/release/python-370/
.. |Maintenance| image:: https://img.shields.io/badge/Maintained%3F-yes-green.svg?style=for-the-badge
   :target: https://GitHub.com/Neywiny/merge-sort/graphs/commit-activity
.. |Open Source Love png2| image:: https://badges.frapsoft.com/os/v2/open-source.png?v=103
   :target: https://github.com/ellerbrock/open-source-badges/
.. |rtd-badge| image:: https://readthedocs.org/projects/merge-sort/badge/?version=latest&style=for-the-badge
   :target: https://merge-sort.readthedocs.io/?badge=latest
.. |licence| image:: https://img.shields.io/badge/License-CC0%201.0-lightgrey.svg
   :target: http://creativecommons.org/publicdomain/zero/1.0/