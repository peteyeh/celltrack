.. image:: https://mybinder.org/badge_logo.svg
 :target: https://mybinder.org/v2/gh/peteyeh/celltrack/HEAD

********************
Quest Image Pipeline
********************

This repository contains an cell image processing pipeline which can be run within Quest, Northwestern's HPC cluster. Briefly, using image stacks of cell lines as input, this pipeline will generate mask images using thresholding, extract feature data from within each mask region, label each region as a cell class, and generate analysis data to show class trends.


1. Cloning the GitHub Repository and Initializing Quest
#######################################################
First, we need to clone the github repository into a directory within Quest::

    git clone https://github.com/peteyeh/celltrack


Next we need to install libaries/dependencies that Quest will need to run our Python scripts. A handy `script <https://github.com/peteyeh/celltrack/blob/master/reference/quest_initialize.sh>`_ is included within this repository to automate this process using Anaconda, which is `pre-installed on Quest <https://kb.northwestern.edu/page.php?id=78623>`_::

    cd reference
    bash quest_initialize.sh


2. Organizing Input and Output Directories
##########################################
In order for Quest jobs to run smoothly we need to ensure that our input and output directories are set up properly.

Output Directory
****************
The intended output directory is hard-coded into template scripts in the reference directory (see `reference/base_mc_fe.sh <https://github.com/peteyeh/celltrack/blob/master/reference/base_mc_fe.sh>`_, `reference/base_km.sh <https://github.com/peteyeh/celltrack/blob/master/reference/base_km.sh>`_, and `reference/base_kp_ca_ci.sh <https://github.com/peteyeh/celltrack/blob/master/reference/base_kp_ca_ci.sh>`_). For example, you should see a line with the output directory ``/projects/p31689/Image_Analysis/output`` specified::

    output="/projects/p31689/Image_Analysis/output"

This output directory can be changed to any arbitrary directory within Quest, but most importantly the output directory must have sufficient disk space to store output files. The size of the output is dependent on how large each image is and how many regions are found within each image, so it is recommended that you test the pipeline on a few image stacks first to estimate output size before submitting many jobs in parallel.

Quest Allocation
****************
While we're here, we should ensure that the allocation (by default ``p31689``) used to submit Quest jobs is up to date in each template script::

    #SBATCH --account=p31689

Input Directory
***************
Similar to the output directory, the locations of the input files are arbitrary. Importantly, however, each image stack must be stored as a single TIFF stack as our code will use ``cv2.imreadmulti`` to read in these files.

The location of each image stack that we wish to submit to Quest should be specified in `src/manifest.txt <https://github.com/peteyeh/celltrack/blob/master/src/manifest.txt>`_. Each line within this manifest file will be submitted to Quest as an independent job, using a single command. Again, it is recommended that you test this pipeline with a small manifest initially, as each job that is submitted to Quest will increase queuing time for your allocation.


3. Submitting Quest Jobs
########################
We now should have everything we need to submit jobs to Quest. Our pipeline currently exists as a two-step process, the mask creation and feature extraction step followed by the k-means classification and analysis step. Note that if this is your first time running this pipeline, you likely will want to train a k-means model and should skip ahead and preview the "K-Means Initialization" step.

Mask Creation and Feature Extraction
**********************************
Once the ``manifest.txt`` file is specified we can execute a simple command::

    bash extract_features.sh

This will initialize one job per image stack. Quest will distribute the jobs across nodes in parallel, and each job will sequentially generate mask images and extract features from the regions of interest. This will generate in the pre-specified output directory a named directory (derived from the image stack name) and a subdirectory (with the timestamp of the latest run) containing individual mask images as well as a CSV with extracted feature data.

Note that this step is the most computationally intensive one - each job is expected to take approximately two hours, but this is highly dependent on the size of each stack and the number of regions we are extracting data from.

K-Means Initialization *[if a k-means model has not yet been trained]*
**********************************************************************
This step should ideally only be run once, to initialize a trained k-means model which will be used to classify all subsequent image stacks. ``manifest.txt`` should be updated to include 1+ representative image stacks, and ``extract_features.sh`` should be run as above on these image stacks if not already. Then, we can execute::

    bash initialize_kmeans.sh

Note that this script is unlike the others, as it will submit **one** job to Quest that aggregates all images from all image stacks to train a single model. Thus, ideally we should be selective about the number of image stacks to include (~5 is probably reasonable) as too many will result in prohibitively long runtimes.

K-Means Classification and Analysis
***********************************

The next step can be run immediately after the first, without any modification to ``manifest.txt``, **assuming that the first set of jobs have all been completed**. That is, you should check ``squeue`` to ensure that there are no pending/running/stuck jobs. We submit the next set of jobs using the following command::

    bash process_features.sh
    
These jobs will add to the existing output directories the following:

#. An updated CSV, with a column of class labels added to the feature data.
#. A set of contoured images, coloured to match each region's label.
#. An ``analysis.csv`` file, which contains analysis data on class counts and areas.

Lastly, these jobs will automatically purge intermediary and redundant files (i.e. the original CSV and pickled data) to conserve disk space.


4. Packaging Output Files
#########################
At this point you should check ``squeue`` to ensure that all jobs have completed successfully. If so, all of the output data of interest should exist in subdirectories separated by image stack. We've included two convenient scripts to merge this data for easy transfer out of Quest::

    bash package_output.sh

or

::

    bash package_output_light.sh

``package_output.sh`` will package all subdirectories from the latest run into ``[output directory]/packaged_output/[timestamp]``.  ``package_output_light.sh`` (which sends the data to ``[output directory]/packaged_output_light/[timestamp]``) will omit some of the bulkier files, namely the labeled feature data and the contoured images.
