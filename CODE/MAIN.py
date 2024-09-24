#==================================================================
#MAIN
#==================================================================

#SETUP CLASSES, DEFINITIONS, AND METHODS
#==================================================================

#Import sys here rather than in EXTERNAL.py, as needed to parse input arguments
import sys

#Obtain the configuration file and version number from input variables
configFileName = sys.argv[1]
dir_tmp = sys.argv[2]
try: versionNum = sys.argv[3]
except: versionNum = 'N/A'

#Load in chosen configuration options; must be done first to set global variables for certain functions/methods correctly
exec(open(configFileName, encoding='utf-8').read())

#Import and setup external libraries
exec(open("./CODE/EXTERNAL.py", encoding='utf-8').read())

#Setup any configuration-derived global variables
exec(open("./CODE/DERIVED.py", encoding='utf-8').read())

#Setup aesthetics
exec(open("./CODE/AESTHETICS.py", encoding='utf-8').read())

#Setup internal directories and naming conventions
exec(open("./CODE/INTERNAL.py", encoding='utf-8').read())

#Configure computation resources
exec(open("./CODE/COMPUTE.py", encoding='utf-8').read())

#Setup classifier model definitions
if classifierModel == 'original': exec(open("./CODE/MODEL_CLASS_ORIGINAL.py", encoding='utf-8').read())
elif classifierModel == 'updated': exec(open("./CODE/MODEL_CLASS_UPDATED.py", encoding='utf-8').read())
else: sys.exit('\nError - Unknown classifierModel specified.')

#Setup reconstructor model definitions
exec(open("./CODE/MODEL_RECON.py", encoding='utf-8').read())

#Import local and remote method/class definitions as needed
exec(open("./CODE/DEFINITIONS.py", encoding='utf-8').read())
exec(open("./CODE/REMOTE.py", encoding='utf-8').read())

#Overwrite any definitions as configured
if overWriteFile != None: exec(open(overWriteFile, encoding='utf-8').read())


#PERFORM OPERATIONAL ROUTINES
#==================================================================

#Train, evaluate, and export a classifier model
exec(open("./CODE/RUN_CLASS.py", encoding='utf-8').read())

#Train, evaluate, and export a reconstruction model
exec(open("./CODE/RUN_RECON.py", encoding='utf-8').read())

#Simulate sampling using pre-trained classifier and reconstruction models
exec(open("./CODE/RUN_RANDS.py", encoding='utf-8').read())


#SHUTDOWN
#==================================================================

#Shutdown ray
if parallelization: 
    _ = ray.shutdown()
    rayUp=False

#Copy the results folder, the config file and ray log directory into it if applicable
sectionTitle('DUPLICATING RESULTS FOR CONFIGURATION')
_ = shutil.copytree(dir_results, dir_results_final)
_ = shutil.copy(configFileName, dir_results_final+'/'+os.path.basename(configFileName))
if debugMode: _ = shutil.copytree(dir_tmp, dir_results_final+'/TMP')

#Notate the completion of intended operations
sectionTitle('CONFIGURATION COMPLETE')

#Make sure this process is closed
exit()
