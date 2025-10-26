#==================================================================
#EXTERNAL
#==================================================================

#ENVIRONMENTAL VARIABLES
#==================================================================

#Check if code is being executed in a jupyter notebook
from IPython import get_ipython
if get_ipython().__class__.__name__ == 'ZMQInteractiveShell': jupyterNotebook = True
else: jupyterNotebook = False

#Dictionary to store environmental variables; passes to ray workers
environmentalVariables = {}

#Setup deterministic behavior for CUDA; may change in future versions
#"Set a debug environment variable CUBLAS_WORKSPACE_CONFIG to :16:8 (may limit overall performance) 
#or :4096:8 (will increase library footprint in GPU memory by approximately 24MiB)."
if manualSeedValue != -1: environmentalVariables["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

#Set matplotlib backend to render plots in the background, unless operating inside of a notebook
if not jupyterNotebook: environmentalVariables["MPLBACKEND"] = "agg"

#Increase memory usage threshold for Ray from the default 95%
environmentalVariables["RAY_memory_usage_threshold"] = "0.99"

#Raise the maximum image size for opencv; note that this can allow for decompression bomb DOS attacks if an untrusted image ends up as an input
environmentalVariables["OPENCV_IO_MAX_IMAGE_PIXELS"] = pow(2,40).__str__()

#Disable Ray memory monitor as it will sometimes decide to kill processes with suprising/unexpected/unmanageable/untracable errors
#environmentalVariables["RAY_DISABLE_MEMORY_MONITOR"] = "1"

if debugMode: 

    #Ray deduplicates logs by default; sometimes verbose output is needed in debug
    environmentalVariables["RAY_DEDUP_LOGS"] = "0"
    
else:
    
    #Stop Ray from crashing the program when errors occur (otherwise may crash despite being handled by try/catch!)
    environmentalVariables["RAY_IGNORE_UNHANDLED_ERRORS"] = "1"
    
    #Change Ray filter level
    environmentalVariables["RAY_LOG_TO_DRIVER_EVENT_LEVEL"] = "ERROR"

    #Prevent Ray from printing spill logs
    environmentalVariables["RAY_verbose_spill_logs"] = "0"

    #Restrict warning levels to only report errors; ONLY applicable to subprocesses!
    environmentalVariables["PYTHONWARNINGS"] = "ignore"

#Load enivronmental variables into current runtime before other imports
import logging, os, warnings
for var, value in environmentalVariables.items(): os.environ[var] = value

#Setup logging configuration and definitions
exec(open("./CODE/LOGGING.py", encoding='utf-8').read())
setupLogging()

#IMPORTS
#==================================================================

import cupy as cp
import copy
import ctypes
import cv2
import contextlib
import datetime
import gc
import glob
import math
import matplotlib
import matplotlib.pyplot as plt
import natsort
import numpy as np
import pandas as pd
import pickle
import PIL
import PIL.ImageOps
import platform
import psutil
import py7zr
import random
import ray
import re
import requests
import scipy
import skimage
import shutil
import time

from contextlib import nullcontext
from IPython.core.debugger import set_trace as Tracer
from matplotlib import colors
from matplotlib.pyplot import figure
from numba import cuda
from numba import jit
from PIL import Image
from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.image import show_cam_on_image
from ray.util.multiprocessing import Pool
from scipy import signal
from skimage.filters import threshold_otsu
from skimage.metrics import mean_squared_error as compare_MSE
from skimage.metrics import structural_similarity as compare_SSIM
from skimage.metrics import peak_signal_noise_ratio as compare_PSNR
from skimage.transform import resize
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
if jupyterNotebook: from tqdm import tqdm
else: from tqdm.auto import tqdm
from tqdm.auto import tqdm
from xgboost import XGBClassifier

import torch
import torch.nn.functional as F
import torch.nn.parallel
import torchvision.transforms as transforms
from torch import nn
from torch import optim
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, models
from torchvision.transforms import v2
from torchvision.transforms.functional import InterpolationMode

#LIBRARY SETUP
#==================================================================

#Benchmarks algorithms and uses the fastest; only recommended if input sizes are consistent
#Untested for RANDS at this time, but it might be beneficial; need to evaluate performance when able
#torch.backends.cudnn.benchmark = True

#Allow anomaly detection in training a PyTorch model; sometimes needed for debugging
#torch.autograd.set_detect_anomaly(True)

#Setup deterministic behavior for torch (this alone does not affect CUDA-specific operations)
if (manualSeedValue != -1): torch.use_deterministic_algorithms(True, warn_only=False)

#Turn off image size checking for pillow; note that this can allow for decompression bomb DOS attacks if an untrusted image ends up as an input
Image.MAX_IMAGE_PIXELS = None

#OS SPECIFIC
#==================================================================

#Determine system operating system
systemOS = platform.system()

#Operating system specific imports 
if systemOS == 'Windows':
    from ctypes import windll, create_string_buffer
    import struct

#==================================================================

#COMPUTATION ENVIRONMENT
#==================================================================

#Store string of all system GPUs (Ray hides them)
systemGPUs = ", ".join(map(str, [*range(torch.cuda.device_count())]))

#Note GPUs available/specified
if not torch.cuda.is_available(): gpus = []
if (len(gpus) > 0) and (gpus[0] == -1): gpus = [*range(torch.cuda.device_count())]
numGPUs = len(gpus)

#Detect logical and physical core counts, determining if hyperthreading is active
logicalCountCPU = psutil.cpu_count(logical=True)
physicalCountCPU = psutil.cpu_count(logical=False)
hyperthreading = logicalCountCPU > physicalCountCPU

#Set parallel CPU usage limit, disabling if there is only one thread remaining
#Ray documentation indicates num_cpus should be out of the number of logical cores/threads
#In practice, specifying a number closer to, or just below, the count of physical cores maximizes performance
#Any ray.remote calls need to specify num_cpus to set environmental OMP_NUM_THREADS variable correctly
if parallelization: 
    if availableThreads==0: numberCPUS = physicalCountCPU
    else: numberCPUS = availableThreads
    if numberCPUS <= 1: parallelization = False
if not parallelization: numberCPUS = 1

#Set number of workers for processing based on available physical cores
if numWorkers_patches <= -1: numWorkers_patches = numberCPUS
elif numWorkers_patches > numberCPUS: numWorkers_patches = numberCPUS
if numWorkers_WSI <= -1: numWorkers_WSI = numberCPUS
elif numWorkers_WSI > numberCPUS: numWorkers_WSI = numberCPUS

#==================================================================
