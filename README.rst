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
We now should have everything we need to submit jobs to Quest. Our pipeline currently exists as a two-step process, the mask creation and feature extraction step followed by the k-means classification and analysis step.

Mask Creation and Feature Analysis
**********************************

K-Means Initialization
**********************

K-Means Classification and Analysis
***********************************


4. Packaging Output Files
#########################
