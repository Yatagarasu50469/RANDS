#==================================================================
#COMPUTE
#==================================================================

#Routine to clean up RAM 
def cleanup(): 
    gc.collect()
    if systemOS != 'Windows': _ = ctypes.CDLL("libc.so.6").malloc_trim(0)

#Reset ray memory and compute; shouldn't be needed but is since pools do not seem to close and free memory properly (set log_to_driver=False to stop all PID messages)
#If not in debug mode, prevent output during shutdown in case of issues with spill objects, as these are manually handled
#If not in debug mode, prevent output during startup in case of issues with port conflict (https://github.com/ray-project/ray/issues/27736 https://github.com/ray-project/ray/issues/18053)
def resetRay(numberCPUS):
    count = 0
    if ray.is_initialized(): rayUp = True
    else: rayUp = False
    while rayUp:
        try: 
            with suppressSTD() if not debugMode else nullcontext(): _ = ray.shutdown()
        except: 
            count += 1
        if count >= 3: print('\nWarning - Ray failed to shutdown correctly, if this message repeatedly appears sequentially, exit the program with CTRL+c.')
        if not ray.is_initialized(): rayUp = False
    _ = gc.collect()
    while not rayUp:
        count = 0
        try: 
            with suppressSTD() if not debugMode else nullcontext(): _ = ray.init(num_cpus=numberCPUS, logging_level=logging.root.level, runtime_env={"env_vars": environmentalVariables}, include_dashboard=False, _temp_dir=dir_tmp)
        except: 
            count += 1
        if count >= 3: print('\nWarning - Ray failed to startup correctly, if this message repeatedly appears sequentially, exit the program with CTRL+c.')
        if ray.is_initialized(): rayUp = True

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
