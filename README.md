<p align="center">
  <img src='/CODE/OTHER/HEADER.PNG' height='321'>
</p>

***
# PROGRAM
<pre>
<b>NAME:</b>           RANDS
<b>MODIFIED:</b>       21 October 2025
<b>VERSION:</b>        0.0.6
<b>LICENSE:</b>        GNU General Public License v3.0
<b>DESCRIPTION:</b>    Risk Assessment Network for Dynamic Sampling
<b>FUNDING:</b>        Development of RANDS has been funded by and developed for NIH Grant 5R01EB033806
	
<b>AUTHOR(S):</b>      David Helminiak    EECE, Marquette University,       Framework, Classifier, RANDS
                Tyrell To          EECE, Marquette University,       Original Classifier (See PUBLICATIONS)

<b>ADVISOR(S):</b>     Dong Hye Ye        COSC, Georgia State University    
                Bing Yu            BIEN, Marquette University
    
</pre>
***

# README CONTENTS
<pre>
<b>PROGRAM DIRECTORY/FILE STRUCTURE</b>
<b>INSTALLATION</b>
  Windows (Native) Pre-Installation
  Docker Pre-Installation
  Ubuntu 20.04, Docker, and WSL2 (Windows Subsystem for Linux) Pre-Installation
  Main Installation
<b>TRAINING/TESTING PROCEDURE</b>
  Configuration
  Running Program
  Results
<b>FAQ</b>
  Program Operation
  Results
  Development
<b>PUBLICATIONS</b>
</pre>

# PROGRAM DIRECTORY/FILE STRUCTURE

**Note:**  *INPUT_* contents are not generally intended to be altered, *OUTPUT_* contents are expected to change according to configured tasks.  

<pre>
----->MAIN_DIRECTORY                          #Complete program contents
  |----->README.md                            #Program documentation
  |----->CHANGELOG.md                         #Versioning and currently anticipated development order
  |----->RANDS.py                             #Program startup; initializes sequential run of CONFIG_# files in subprocesses
  |----->CONFIG_#-description.py              #Configuration for a single program run
  |----->CODE                                 #Location to store any code used in the course of running the program
  |  |----->AESTHETICS.py                     #Handles UI elements at runtime
  |  |----->CONFIGURATION.py                  #Default configuration parameters to be overridden by CONFIG_#-description.py file(s)
  |  |----->COMPUTE.py                        #Prepare computation CPU/GPU/RAM environment
  |  |----->DEFINITIONS.py                    #Global methods
  |  |----->EXTERNAL.py                       #Environmental variables and setup of third-party libraries
  |  |----->INTERNAL.py                       #Directory and file handling
  |  |----->MAIN.py                           #Performs program setup, operations, and shutdown
  |  |----->MODEL_RECON.py                    #Reconstruction model
  |  |----->MODEL_CLASS_ORIGINAL.py           #Classification model, re-implemented from original work
  |  |----->MODEL_CLASS_UPDATED.py            #Classification model, new/updated approach
  |  |----->REMOTE.py                         #Parallel actors and methods
  |  |----->RUN_CLASSIFIER.py                 #Train/Evaluate classifier model
  |  |----->RUN_RANDS.py                      #Simulate Risk Assessment Network for Dynamic Sampling
  |  |----->RUN_RECON.py                      #Train/Evaluate reconstruction model
  |  |----->OTHER                             #Files not directly used in program execution
  |  |----->SCRIPTS                           #Jupyter notebook files intended for one-time use and/or development
  |  |  |----->patchDeeperExtraction.ipynb    #Locate and label spatial data for extracted patch images
  |  |  |----->patchSegmentationTest.ipynb    #Visualize impact of parameter variation on the selection of patches from WSI
  |  |  |----->ARCHIVE                        #Development files no longer used, but to be retained for reference/documentation
  |----->DATA                                 #Data directory
  |  |----->PATCHES                           #Training/evalutation data for patch classification network
  |  |  |----->INPUT_PATCHES                  #Labeled patches; extracted from the exact images in INPUT_WSI
  |  |  |  |-----><i>sample#</i>_<i>side#</i>               #Directory corresponding to originating WSI
  |  |  |  |  |----->PS<i>patchData</i>.tif          #Lossless patch image(s); <i>patchData</i> : <i>sample#</i>_<i>side#</i>_<i>patchID#</i>_<i>row#</i>_<i>column#</i>
  |  |  |  |----->metadata_patches.csv        #Patch metadata for every lossless image in the directory
  |  |  |----->INPUT_WSI                      #Labeled WSI; the exact images INPUT_PATCHES data was extracted from
  |  |  |  |-----><i>sample#</i>_<i>side#</i>.jpg           #Stitched and BaSIC-processed (not color-corrected) WSI
  |  |  |----->OUTPUT_FEATURES                #Resultant feature data generated for patch images
  |  |  |----->OUTPUT_SALIENCY_MAPS           #Resultant saliency map (weight data) generated for patch images
  |  |  |----->OUTPUT_VISUALS                 #Visualizations produced in patch image classification tasks
  |  |  |  |----->FUSION_GRIDS                #Fusion predictions visualized in grid form
  |  |  |  |----->LABEL_GRIDS                 #Ground-truth labels spatially visualized in grid form (Green->Benign, Red->Malignant)
  |  |  |  |----->OVERLAID_FUSION_GRIDS       #Fusion predictions visualized on WSI
  |  |  |  |----->OVERLAID_LABEL_GRIDS        #Label grids visualized on WSI
  |  |  |  |----->OVERLAID_PREDICTION_GRIDS   #Predictions visualized on WSI
  |  |  |  |----->OVERLAID_SALIENCY_MAPS      #Saliency maps upsampled with bilinear interpolation and visualized on WSI
  |  |  |  |----->PREDICTION_GRIDS            #Predictions visually visualized in grid form
  |  |  |  |----->SALIENCY_MAPS               #Saliency maps spatially visualized at original dimensions (Brighter->Critical)
  |  |----->RECON                             #Training/evaluation data for reconstruction model (all WSI, including those used for training the classifier)
  |  |  |----->INPUT_WSI                      #WSI not used for training the classifier for use in the reconstruction model
  |  |  |  |-----><i>sample#</i>_<i>side#</i>.jpg           #Stitched and BaSIC-processed (not color-corrected) WSI
  |  |  |----->OUTPUT_PATCHES                 #Patch images/data extracted from WSI
  |  |  |----->OUTPUT_FEATURES                #Resultant feature data generated for extracted patch images
  |  |  |----->OUTPUT_INPUT_DATA              #Resultant classifier data to be used as input data for the reconstruction model
  |  |  |----->OUTPUT_SALIENCY_MAPS           #Resultant saliency map (weight data) generated for extracted patch images
  |  |  |----->OUTPUT_VISUALS                 #Visualizations produced in classification tasks
  |  |  |  |----->FUSION_GRIDS                #Fusion predictions visualized in grid form
  |  |  |  |----->INPUT_DATA                  #Classification model predictions; visualizations of input data for reconstruction model
  |  |  |  |----->OVERLAID_FUSION_GRIDS       #Fusion predictions visualized on WSI
  |  |  |  |----->OVERLAID_PREDICTION_GRIDS   #Predictions visualized on WSI
  |  |  |  |----->OVERLAID_SALIENCY_MAPS      #Saliency maps upsampled with bilinear interpolation and visualized on WSI
  |  |  |  |----->PREDICTION_GRIDS            #Predictions visually visualized in grid form
  |  |  |  |----->SALIENCY_MAPS               #Saliency maps spatially visualized at original dimensions (Brighter->Critical)
  |----->RESULTS                              #Results directory
  |  |----->PATCHES                           #Patch classification training/evaluation results
  |  |  |----->VISUALS                        #Generated visuals for classification of patches and their WSI
  |  |  |  |----->FUSION_GRIDS                #Fusion predictions visualized in grid form
  |  |  |  |----->OVERLAID_FUSION_GRIDS       #Fusion predictions visualized on WSI
  |  |  |  |----->OVERLAID_PREDICTION_GRIDS   #Label grids visualized on WSI
  |  |  |  |----->PREDICTION_GRIDS            #Predicted labels spatially visualized in grid form (Green->Benign, Red->Malignant)
  |  |  |----->classReport_<i>dataType</i>.csv       #Classification report of common metrics; <i>dataType</i> : <i>patches/WSI</i>_<i>initial/fusion</i>
  |  |  |----->confusionMatrix_<i>dataType</i>.tif   #Confusion matrix visualization; <i>dataType</i> : <i>patches/WSI</i>_<i>initial/fusion</i>
  |  |  |----->predictions_<i>dataType</i>.csv       #Predicted per-patch/WSI labels; <i>dataType</i> : <i>patches/WSI</i>_<i>initial/fusion</i>
  |  |  |----->summaryReport_<i>dataType</i>.csv     #Summary of additional result metrics;  <i>dataType</i> : <i>patches/WSI</i>_<i>initial/fusion</i>
  |  |----->MODELS                            #Models exported by the program in ONNX (C# compatible) and more python-compatible forms
  |  |----->RECON                             #Reconstruction model results
  |  |  |----->TRAIN                          #Training metrics/visualizations
  |  |  |----->TEST                           #Testing metrics/visualizations
  |  |----->WSI                               #WSI classification evaluation results
  |  |  |----->LABEL_GRIDS                    #Predicted labels spatially visualized in grid form (Green->Benign, Red->Malignant)
  |  |  |----->OVERLAID_LABEL_GRIDS           #Label grids visualized on WSI
  |  |  |----->predictions_WSI_<i>dataType</i>.csv   #Predicted WSI labels; <i>dataType</i> : <i>initial/fusion</i>
  |  |  |----->classReport_WSI_<i>dataType</i>.csv   #Classification report of common metrics; <i>dataType</i> : <i>initial/fusion</i>
  |  |  |----->confusionMatrix_<i>dataType</i>.tif   #Confusion matrix visualization; <i>dataType</i> : <i>initial/fusion</i>
  |  |  |----->summaryReport_WSI_<i>dataType</i>.csv #Summary of additional result metrics;  <i>dataType</i> : <i>initial/fusion</i>
</pre>

# INSTALLATION

**Note:** Throughout this document the '$ ' prefix is used to denote new lines and are not intended to be copied/executed.

Follow the instructions provided in the pre-installation guide specific to your system's operating system followed by those in the **Main Installation** section. The package versions do not necessarily need to match with those listed. However, should the program produce unexpected errors, installing a specific version of a package might be able to resolve the issue. Note that the multiplierz pacakage, must be installed from the provided repository fork, as specified in the installation guide for some methods to work properly; see **FAQ** for further information. 

**Software/Package Combinations**  
	
    Python             3.10.12
    pip                24.2
    NVIDIA Driver      560.81
	
    cupy-cuda12x       13.3.0
    datetime           5.5
    glob2              0.7
	grad-cam           1.5.3
    ipython            8.26.0
	ipywidgets         8.1.3
    joblib             1.4.2
    matplotlib         3.9.1
    multivolumefile    0.2.3
    natsort            8.4.0
    numba              0.60.0
    numpy              1.26.4
    opencv-python      4.10.0.84
	openpyxl           3.1.5
    pandas             2.2.2
    pathlib            1.0.1
    pillow             10.4.0
    psutil             6.0.0
    py7zr              0.21.1
    ray                2.33.0
    scikit-image       0.22.0
    scikit-learn       1.5.1
    scipy              1.14.0
    torch              2.2.2+cu121
    torchaudio         2.2.2+cu121
    torchvision        0.17.2+cu121
    tqdm               4.66.4
	xgboost            2.1.1

**Minimum Hardware Requirements:** As more functionality is continually being added, minimum hardware specifications cannot be exactly ascertained at this time, however validation of functionality is performed on systems containing 64+ GB DDR3/4/5 RAM, 32+ CPU threads at 3.0+ GHz, A4000/2080Ti/4090 GPUs, and 1TB+ SSD storage. 

**GPU/CUDA Acceleration:** Highly recommended. Note that there shouldn't be a need to manually install the CUDA toolkit, or cudnn as pytorch installation using pip should come include the neccessary files. 

## Windows (Native) Pre-Installation

If not already setup, install the latest Python v3.11 version (https://www.python.org/downloads/) selecting the options to install for all users, and addding python.exe to PATH. Operation using Python v3.12+ has not yet been validated.

Open a command prompt: 

	$ python -m pip install --upgrade pip
	$ pip3 install pywin32

## Docker Pre-Installation

This program can also be installed in a Docker container (confirmed functional on CentOS 7/8). 
OS-specific Docker installation instructions: (https://docs.docker.com/engine/install/) 
For GPU acceleration you will need to first install the NVIDIA container toolkit and configure Docker to use the correct runtime: (https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html#installing-the-nvidia-container-toolkit) 
After installing/entering the Docker container, follow the Ubuntu Pre-Installation instructions.  

The quick commands for initial container setup (shm-size should be set to 50% of available system RAM):
    $ docker run --gpus all -it --shm-size=64gb --runtime=nvidia --name RANDS nvcr.io/nvidia/pytorch:24.05-py3
	$ rm -rf /usr/local/lib/python3.10/dist-packages/cv2
    $ python -m pip install --upgrade pip
    $ pip3 uninstall -y -r <(pip freeze)
	
Additional useful flags for a Docker container setup:

    '-v /mnt/Volume/:/workspace': Map mounted volume on host system to /workspace directory inside of the container; allows transmission of data between container and host
    '-p 8889:8888': Map port 8888 inside of the container to 8889 (change on a per-user basis) on the host network (in case of performing development with jupyter-notebook on systems with multiple users)

## Ubuntu 20.04, Docker, and WSL2 (Windows Subsystem for Linux) Pre-Installation

Open a terminal window and perform the following operations: 
	
    $ sudo apt-get update -y 
	$ sudo apt-get upgrade -y 
    $ sudo apt-get install -y python3-pip build-essential software-properties-common ca-certificates gnupg libgl1-mesa-dev
	$ python3 -m pip install --upgrade pip

## Main Installation

Open a terminal or command prompt (**not as an administrator**) and run the commands shown below. 
    
	$ pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    $ pip3 install opencv-python datetime glob2 IPython pandas pathlib psutil matplotlib numpy numba pillow ray[serve]==2.33.0 scipy scikit-learn natsort scikit-image tqdm py7zr multivolumefile notebook==6.5.6 ipywidgets openpyxl xgboost grad-cam cupy-cuda12x

Packages that may no longer be required (if there is a runtime error that references one of these packages, please notify the repository maintainer): 

    $ pip3 install aiorwlock sobol sobol-seq multiprocess pydot graphviz joblib

***
# TRAINING/TESTING PROCEDURE

## Configuration
**Note:** There are very few sanity checks inside of the code to ensure only correct/valid configurations are used. If an unexpected error occurs, please double check the sample and program configuration files to ensure they are correct before opening an issue or contacting the author. Thank you.

All parameters may be altered in a configuration file (Ex. ./CONFIG_0.py), which override defaults stored in "./CODE/CONFIGURATION.py". Variable descriptions are provided inside of the default configuration file and are grouped according to the following method:

    L0: Tasks to be Performed
    L1: Compute Hardware & Global Methods
    L2: Classification
	|----->L2-1: Cross-Validation
	|----->L2-2: Classification Models
	|  |----->L2-1-1: Original
	|  |----->L2-1-2: Updated
	|----->L2-3: Reconstruction Data Generation
    L3: Reconstruction & Sampling
	|----->L3-1: Training
	|----->L3-2: Architecture
	|----->L3-3: Optimization
    L4: Rarely Changed Parameters
	|----->L4-1: Classification
	|----->L4-2: Reconstruction
	|----->L4-3: Simulation
	L5: Isolated Parameters - Do not change, alternative options not currently functional
    L6: Debug/Deprecated - Likely to be removed in future

Multiple configuration files in the form of CONFIG_*descriptor*.py, can be generated and then automatically run sequentially. RESULTS_*descriptor* folders will correspond with the numbering of the CONFIG files, with the RESULTS folder without a description, containing results from the last run performed. 

## Running Program
After configuration, to run the program perform the following command in the root directory (if on Windows, replace python3 with python):

    $ python3 ./START.py

## Results
In the case that multiple configuration files are provided in the form of: CONFIG_*descriptor*.py, the RESULTS folder will be duplicated with the same _*descriptor* suffix for ease of testing. Configuration file will be copied into the results directory at the termination of the program. 

# FAQ

### I read through the README thoroughly, but I'm still getting an error, am confused about how a feature should work, or would like a feature/option added
Please check if it has already been addressed, or open an issue on the Github repository at https://github.com/Yatagarasu50469/RANDS/issues 

At this time there is no legacy version support; it would be advisable to verify that the latest release is being used and installed packages match with those listed in the corresponding README. 

### Can the data used for training/evaluating this program be shared or made available?
The datasets used for training and evaluating this program are not currently publicly available. 
Requests for permission and access may be directed to Dr. Bing Yu (https://mcw.marquette.edu/biomedical-engineering/directory/bing-yu.php) 

### Can this program be run using MacOS?
This may be possible (portions of the code are known to have been functional at one point on a Hackintosh with an NVIDIA 1080Ti), but verifiable, up-to-date installation instructions are not currently available. 

### Running in WSL2 (Windows Subsystem for Linux), Ray crashes due to insufficient memory
WSL2 is still a virutal machine and does not see/allocate all of available system memory (the current default is half). This may be increased manually and might be sufficient to allow the program to run. Open a command prompt and enter the instruction below, choosing to create a new file if prompted. 

	$ cd %UserProfile%
	$ notepad.exe .wslconfig
	
	#Paste the following into the new notepad window, make changes as appropriate for your system, then save without an extension and exit
	[wsl2]
	memory=64GB
	processors=32

	#Back in command prompt, use the next instruction to remove the .txt extension from .wslconfig
	$ ren .wslconfig.txt .wslconfig

	#Open PowerShell as an administrator and run the following to reboot WSL2
	$ wsl --shutdown
	$ restart-service LxssManager

## Program Operation

### Why aren't all of the system GPUs used during training?
Distributed GPU training is not currently supported. Multiple GPUs can be and are leveraged in simulated testing when more than a single sample is being evaluated. 

### Using multiple GPUs fails with NCCL Errors
Presuming this error occurs on a Linux OS, increase the available shared memory /dev/shm/ to at least 512 MB. If using a Docker container this can be done by first shutting down Docker completely (sudo systemctl stop docker) editing the container's hostconfig.json file (edited with root privileges at /var/lib/docker/containers/containerID/hostconfig.json), changing the ShmSize to 536870912 and then starting docker back up (sudo systemctl start docker). The changed size may be verified with: df -h /dev/shm 

### The program appears to regularly freeze and I can't interact with the computer
Certain sections of the program are extremely compute/memory/storage instensive and are expected to freeze the graphical output of even upper-end hardware; to verify check Task Manager on Windows, htop on Linux-based platforms, and/or nvidia-smi output for utlization levels. 

### The program appears to hang completely
You can try enabling debugMode in the configuration, which may show additional information. The program has been seen in some installations to stop just after startup without output if there is an issue with initializing Ray (typically requires adjusting network parameters or installed package versioning).  

### Error indicates that the GPU should be disabled, settting parallelism to False doesn't resolve this
The parallelism flag in the configuration file does not control GPU usage, only CPU usage. In order to disable GPUs entirely, set availableGPUs = '-1'.

## Results

### When the images are opened, they appear blank or fail to load
The WSI are sufficiently large that many common image viewing programs will fail to open them correctly (someitmes as a security measure against decompression attacks). gThumb (CentOS), Windows Photo Viewer, and GIMP are programs confirmed as being able to load these images. 

## Development

### Encountering OOM errors during new code development
Ray/Python pin objects in memory if any reference to them still exists; references (particularly to large objects) must be prevented or deleted. Admittedly, there's probably a better way of handling this, but the current coding practices for reducing memory overhead and OOM errors are as follows:
1. Delete Ray references when they are no longer needed, then calling cleanup()
2. Reset Ray (resetRay(numberCPUS), which also has been set to call cleanup() after major remote computations and results have been retrieved
3. On returns from remote calls, copy the data to prevent reference storage  
   -if _ = ray.get() (i.e. returning a None object) -> No problem, a reference was not created  
   -if ray.get(), returns list/array -> use ray.get().copy()  
   -if ray.get(), returns something else -> use copy.deepcopy(ray.get())  
4. Delete large objects when they are no longer needed, then calling cleanup()
5. Call cleanup() after major methods return to MAIN.py

***
# PUBLICATIONS
**Note:** DUV-FSM (Deep-Ultraviolet Fluoresence Scanning Microscope) refers to the first generation experimental system, whereas DDSM denotes the second generation variant. 

## RESEARCH PRODUCED USING THIS CODE

**Prior/Original and Vision Transformer Classification Network**   
**Version(s):** v0.0.4  
**Subject:** Breast Cancer Classification for DUV-FSM   
**Citation(s):** P. Afshin et al., ‘Breast Cancer Classification in Deep Ultraviolet Fluorescence Images Using a Patch-Level Vision Transformer Framework’, arXiv preprint arXiv:2505. 07654, 2025.   
**Available:** (https://arxiv.org/pdf/2505.07654?)   
**Note:** Results for comparitive network only; vision transformer network implemented outside of this code and was not yet included herein.   

## RELATED RESEARCH

**Prior/Original Classification Network**    
**Subject:** Breast Cancer Classification for DUV-FSM   
**Available:** (https://github.com/tyrellto/breast-cancer-research/tree/main)   
**Note**: Original classification network code for RANDS was derived from this existing work (published under GNU GPLv3), but **entirely** rewritten. 

**Prior/Original and Diffusion Network**  
**Subject:** Breast Cancer Classification for DUV-FSM   
**Citation(s):** G. S. Salem, T. To, J. Jorns, T. Yen, B. Yu, and D. H. Ye, “Deep learning for automated detection of breast cancer in deep ultraviolet fluorescence images with diffusion probabilistic model,” arXiv (Cornell University), Jul. 2024, doi: https://doi.org/10.1109/isbi56570.2024.10635349.   
**Available:** (https://pubmed.ncbi.nlm.nih.gov/40313564/)   

**Prior/Original Classification Network**    
**Subject:** Breast Cancer Classification for DUV-FSM   
**Citation(s):** T. To et al., “Deep learning classification of deep ultraviolet fluorescence images toward intra-operative margin assessment in breast cancer,” Frontiers in Oncology, vol. 13, Jun. 2023, doi: 10.3389/fonc.2023.1179025   
**Available:** (https://pmc.ncbi.nlm.nih.gov/articles/PMC10313133/)   

**Prior/Original Classification Network**   
**Subject:** Breast Cancer Classification for DUV-FSM   
**Citation(s):** Lu T, Jorns JM, Ye DH, Patton M, Gilat-Schmidt T, Yen T, Yu B. Analysis of Deep Ultraviolet Fluorescence Images for Intraoperative Breast Tumor Margin Assessment. Proc SPIE Int Soc Opt Eng. 2023 Jan-Feb;12368:1236806. doi: 10.1117/12.2649552. Epub 2023 Mar 6. PMID: 37292087; PMCID: PMC10249647.   
**Available:** (https://pmc.ncbi.nlm.nih.gov/articles/PMC10249647/)   

**Prior/Original Classification Network**  
**Subject:** Breast Cancer Classification for DUV-FSM   
**Citation(s):** T. To, “Deep Learning Classification of Deep Ultraviolet Fluorescence Images for Margin Assessment During Breast Cancer Surgery,” Master’s Thesis, Marquette University, 2023.   
**Available:** (https://epublications.marquette.edu/theses_open/768)   

**Texture Analysis Classification**  
**Subject:** Breast Cancer Classification for DUV-FSM   
**Citation(s):** Lu T, Jorns JM, Ye DH, Patton M, Fisher R, Emmrich A, Schmidt TG, Yen T, Yu B. Automated assessment of breast margins in deep ultraviolet fluorescence images using texture analysis. Biomed Opt Express. 2022 Aug 30;13(9):5015-5034. doi: 10.1364/BOE.464547. PMID: 36187258; PMCID: PMC9484420.   
**Available:** (https://pmc.ncbi.nlm.nih.gov/articles/PMC9484420/)   

**Prior/Original Classification Network**  
**Subject:** Breast Cancer Classification for DUV-FSM   
**Citation(s):** T. To, S. H. Gheshlaghi and D. H. Ye, "Deep Learning for Breast Cancer Classification of Deep Ultraviolet Fluorescence Images toward Intra-Operative Margin Assessment," 2022 44th Annual International Conference of the IEEE Engineering in Medicine & Biology Society (EMBC), Glasgow, Scotland, United Kingdom, 2022, pp. 1891-1894, doi: 10.1109/EMBC48229.2022.9871819   
**Available:** (https://doi.org/10.1109/EMBC48229.2022.9871819)

**Experimental DUV-FSM Platform**   
**Subject:** First Generation Experimental Platform for DUV-FSM   
**Citation(s):** Lu T, Jorns JM, Patton M, Fisher R, Emmrich A, Doehring T, Schmidt TG, Ye DH, Yen T, Yu B. Rapid assessment of breast tumor margins using deep ultraviolet fluorescence scanning microscopy. J Biomed Opt. 2020 Nov;25(12):126501. doi: 10.1117/1.JBO.25.12.126501. PMID: 33241673; PMCID: PMC7688317.   
**Available:** (https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7688317/)   
