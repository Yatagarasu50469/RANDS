#====================================================================
#CONFIGURATION
#====================================================================

##################################################################
#L0: TASKS TO BE PERFORMED
##################################################################

#Should classification/cross-validation be performed on patches
classifierTrain = True

#Should classifier model components be saved/exported
classifierExport = True

#Should classifier model components be evaluated
classifierEvaluate = True

#Should training data be generated for the reconstructor
classifierRecon = False

#Should a reconstructor model be trained
#Not currently operational
reconstructorTrain = False

#Should the reconstructor model components be saved/exported
#Not currently operational
reconstructorExport = False

#Should the reconstructor model be evaluated
#Not currently functional
reconstructorEvaluate = False

#Should dynamic sampling be simulated
#Not currently operational
simulateRANDS = False

##################################################################


##################################################################
#L1: COMPUTE HARDWARE & GLOBAL METHODS
##################################################################

#How many samples should be submitted in a batch through pytorch models used in classifier
#Incrementing in powers of 2 recommended to best leverage common GPU hardware designs
#For ResNet and DenseNet a 2080TI 11GB can handle 64x3x224x224 (resizeSize=224) or 16x3x400x400 (resizeSize=0)
#In some cases, the GPU memory may not be fully released (even when explicitly told to do so), so using a lower batch size may help prevent OOM
batchsizeClassifier = 32

#Which GPU(s) devices should be used (last specified used for training); (default: [-1], any/all available; CPU only: [])
#Currently limited to a single GPU for training and inferencing
gpus = [-1]

#Should parallelization calls be used to leverage multithreading where able
parallelization = True

#If parallelization is enabled, how many CPU threads should be used? (0 will use any/all available)
#Recommend starting at half of the available system threads if using hyperthreading,
#or 1-2 less than the number of system CPU cores if not using hyperthreading.
#Adjust to where the CPU just below 100% usage during parallel operations 
availableThreads = 0

##################################################################


##################################################################
#L2: CLASSIFICATION
##################################################################

#Which classifier model file should be loaded: 'original' or 'updated' (default: original)
classifierModel = 'original'

#What is the camera resolution in mm/pixel for the instrument that acquired the data being used
#This value is arbitary and has not been certified/determined
cameraResolution = 0.001

#What is the minimum area/quantity (in mm^2) of foreground data that should qualify a patch for classification
#Decrease for increased sensitivity and vice versa; result should not exceed (patchSize*cameraResolution)**2
#As the classifier was not trained to handle blank background patches, setting too low will harm performance
#This value is arbitary and has not been certified/determined
minimumForegroundArea = 0.283**2

#Minimum value [0, 255] for a grayscale pixel to be considered as a foreground location during patch extraction (default: 11)
#-1 will automatically determine a new value as the minimum Otsu threshold across all available WSI; default value from prior determination
patchBackgroundValue = 11

#==================================================================
#L2-1: PATCHES
#==================================================================

#******************************************************************
#L2-1-1: ORIGINAL CLASSIFICATION MODEL
#******************************************************************

#Should features be extracted for patches and overwrite previously generated files
overwrite_patches_features = True

#Should saliency maps be determined for patches and overwrite previously generated files
overwrite_patches_saliencyMaps = True

#Should the decision fusion mode be used for patch classification (default: True)
fusionMode_patches = True

#Should saliency maps and their overlays be visualized for patch WSI
visualizeSaliencyMaps_patches = True

#Should label grids and their overlays be visualized for patch WSI; will overwrite previously generated files
#Files should be updated if thresholdWSI is changed
visualizeLabelGrids_patches = True

#Should prediction grids and their overlays be visualized for patch WSI
visualizePredictionGrids_patches = True

#What ratio of malignant to benign patches should be used to label a whole WSI as malignant (default: 0)
#When set to 0, will label a WSI as malignant if any one component patch has a malignant label
thresholdWSI_label = 0

#What ratio of malignant to benign patches should be used to predict a whole WSI as malignant (default: 0.15)
#When set to 0, will label a WSI as malignant if any one patch is predicted as malignant
#Unknown what the original work used for this value, but for original work replication, a value of 0.15 seems appropriate 
#Replication of original results occurs with values between 0.12-0.19
#If this value is changed, then should enable visualizeLabelGrids_patches to update stored data
thresholdWSI_prediction = 0.15

#If folds for XGB classifier cross validation should be manually defined (e.g. [['S1', 'S3'], ['S4', 'S2']]), else use specify number of folds to generate
#Default matches folds used in prior work (https://doi.org/10.3389/fonc.2023.1179025)
#Omits 6 available samples, with folds holding: 11, 12, 12, 12, 13 samples respectively; this may have been to better balance class distribution
#Presently, all available (non-excluded) samples (not just those in manualFolds) are currently used for training the exported/utilized final classifier
manualFolds = [['2_1', '9_3', '11_3', '16_3', '34_1', '36_2', '40_2', '54_2', '57_2', '60_1', '62_1'],
               ['17_5', '20_3', '23_3', '24_2', '28_2', '30_2', '33_3', '51_2', '52_2', '59_2', '63_3', '66_2'], 
               ['12_1', '14_2', '22_3', '26_3', '35_4', '44_1', '45_1', '47_2', '49_1', '53_2', '56_2', '68_1'], 
               ['4_4', '5_3', '8_1', '10_3', '25_3', '27_1', '29_2', '37_1', '42_3', '48_3', '50_1', '69_1'], 
               ['7_2', '15_4', '19_2', '31_1', '43_1', '46_2', '55_2', '58_2', '61_1', '64_1', '65_1', '67_1', '70_1']]

#Which pre-trained weight sets should be used for ResNet and DenseNet model: 'IMAGENET1K_V1', 'IMAGENET1K_V2'
#Unclear which weights were used for TensorFlow ResNet50 variant in original implementation, but V2 did improve scores a bit when using resizeSize = 0
weightsResNet = 'IMAGENET1K_V2'

#Which pre-trained weight sets should be used for DenseNet model: 'IMAGENET1K_V1'
weightsDenseNet = 'IMAGENET1K_V1'

#******************************************************************

#******************************************************************
#L2-1-2: UPDATED CLASSIFICATION MODEL
#******************************************************************
#TBD
#******************************************************************

#==================================================================
#L2-2: RECONSTRUCTION DATA GENERATION
#==================================================================

#Should WSI preparation and patch extraction (for patch-specific WSI) overwrite previously generated files
overwrite_recon_patches = True

#Should features be extracted for WSI extracted patches (for patch-specific WSI) and overwrite previously generated files
overwrite_recon_features = True

#Should saliency maps be determined for WSI extracted patches (for patch-specific WSI) and overwrite previously generated files
overwrite_recon_saliencyMaps = True

#Should the decision fusion mode be used for patch classification (default: True)
fusionMode_recon = True

#Should visuals of the reconstruction model input data be generated
visualizeInputData_recon = True

#Should saliency maps and their overlays be visualized for patch WSI
visualizeSaliencyMaps_recon = True

#Should prediction grids and their overlays be visualized for patch WSI
visualizePredictionGrids_recon = True

#==================================================================

###########################################################################################################



###########################################################################################################
#L3: RECONSTRUCTION & SAMPLING
#NOT CURRENTLY OPERATIONAL
###########################################################################################################

#==================================================================
#L3-1: TRAINING/VALIDATION DATA PROCESSING
#==================================================================

#Should input data be padded for even dimensions throughout up and down sampling (default: True)
#If this and/or the value of processInputData changes, then training data generation will need to be rerun
recon_padInputData = True

#How should input data be processed for model input: 'standardize', 'normalize', or 'None' (default: 'None')
#If this and/or the value of padInputData changes, then training data generation will need to be rerun
recon_processInputData = 'standardize'

#When generating training and validation data, what scanning method should be used: 'random', or 'pointwise' (default: 'random')
recon_scanMethodTV = 'random'

#What percentage of points should be initially acquired (random) during training and c value optimization (default: 1)
recon_initialPercToScanTrain = 1

#Stopping percentage for number of acquired pixels for training (default: 100)
recon_stopPercTrain = 100

#Percentage of points (group-based) to acquire each iteration during training/validation data generation (default: None will scan only one location per iteration)
recon_percToScanTV = 1

#Percentage of points to acquire between visualizations (checkpoints); if all steps should be, then set to None (pointwise default: 1; linewise default: None)
recon_percToVizTV = None

#What percentage of the training data should be used for training (default: 0.8)
#1.0 or using only one input sample will use training loss for early stopping criteria
recon_trainingSplit = 0.8

#==================================================================

#==================================================================
#L3-1: TRAINING
#==================================================================

#Should the training data be augmented at the end of each epoch (default: True)
recon_augTrainData = True

#Should visualizations of the training progression be generated (default: True)
recon_trainingProgressionVisuals = True

#How often (epochs) should visualizations of the training progression be generated (default: 10)
recon_trainingVizSteps = 10

#==================================================================

#==================================================================
#L3-2: ARCHITECTURE
#==================================================================

#Reference number of how many convolutional filters to use in building the model
recon_numStartFilters = 64

#How many layers down should the network go
recon_networkDepth = 4

#Should the number of filters double at depth
recon_doubleFilters = True

#What initialization should be used: 'xavier_uniform', 'xavier_normal', 'kaiming_uniform', 'kaiming_normal' (default: 'xavier_uniform')
recon_initialization='xavier_uniform'

#Activations for network sections (input, down, embedding down, upsampling, embedding up, final): 'leaky_relu', 'relu', 'prelu', 'linear'
recon_inAct, recon_dnAct, recon_upAct, recon_fnAct = 'leaky_relu', 'leaky_relu', 'relu', 'relu'

#Negative slope for any 'leaky_relu' or 'prelu' activations (default: 0.2)
recon_leakySlope = 0.2

#Should bias be enabled in convolutional layers (default: True)
recon_useBias = True

#Binomial blur padding model mode: 'reflection', 'zero', 'partialConvolution'
recon_blurPaddingMode = 'reflection'

#Sigma for binomial filter applied during upsampling; 0 to disable (default: 3)
recon_sigmaUp = 1

#Sigma for binomial filter applied during downsampling; 0 to disable (default: 1)
recon_sigmaDn = 0

#Should instance normalization be used throughout the network (default: False)
recon_dataNormalize = False

#==================================================================

#==================================================================
#L3-3: OPTIMIZATION
#==================================================================

#Which optimizer should be used: 'AdamW', 'NAdamW', 'Adam', 'NAdam', 'RMSprop', or 'SGD' (default: 'NAdamW')
recon_optimizer = 'NAdamW'

#What should the initial learning rate for the model optimizer(s) be (default: 1e-4)
recon_learningRate = 1e-4

#What loss fucntion should be used for training the model: 'MAE' or 'MSE' (default: 'MAE')
recon_lossFunction = 'MAE'

#How many epochs should the model training wait to see an improvement using the early stopping criteria (default: 100)
recon_maxPatience = 100

#How many epochs should a model be allowed to train for (default: 1000)
recon_maxEpochs = 1000

#If cosine annealing with warm restarts should be used (default: True)
recon_scheduler_CAWR = True

#If using cosine annealing, what should the period be between resets (default: 1)
recon_schedPeriod = 1

#If using cosine annealing, how should the period length be multiplied at each reset (default: 1)
recon_schedMult = 1

#Across how many epochs across should two moving averages be considered for early stopping criteria (default: 0)
#Historical loss = mean(losses[-sepEpochs*2:-sepEpochs]; Current loss = mean(losses[-sepEpochs:])
#0 will trigger early stopping criteria relative to occurance of best loss
recon_sepEpochs = 0

#How many epochs should a model be allowed to remain wholly stagnant for before training termination (default: 10)
recon_maxStagnation = 10

#==================================================================

###########################################################################################################



###########################################################################################################
#L4: RARELY CHANGED PARAMETERS
###########################################################################################################

#When splitting WSI images, what size should the resulting patches be (default: 400)
#Should remain consistent with patch sizes given for training
patchSize = 400

#Specify what symmetrical dimension patches should be resized to; if no resizing is desired leave as 0 (default: 224)
#Leaving as 0 will increase training time (also must change batchsizeClassifier), but can lead to improved scores
#Original implementation uses 224, though with adaptive average pooling this isn't actually neccessary
resizeSize_patches = 224

#Specify what symmetrical dimension WSI should be resized to when generating saliency maps (default: 224)
#If fusion method were to be further developed, this should be changed to maintain the original sample aspect ratio. 
resizeSize_WSI = 224

#How thick should the grid lines be when generating overlay images (defualt: 50)
gridThickness = 50

#Should saliency map overlays be placed over data converted to grayscale for clearer visualization (default: True)
overlayGray = True

#Should images be saved to lossless (.tif) or compressed (.jpg) image format (default: False)
#Any images that are anticipated to be reused (such as extracted patches), should be hardcoded to be saved in a lossless format by default
exportLossless = False

#For .jpg image outputs, what should the compression quality (%) be (default: 95)
#WARNING: Setting to 100 is not sufficient to generate in lossless/exact outputs; if that is desired, use overlayLossless instead! 
exportQuality = 95

#What weight should be used when overlaying data
overlayWeight = 0.5

#RNG seed value to help ensure run-to-run consistency (-1 to disable)
manualSeedValue = 0

#Should warnings and info messages be shown during operation (default: False)
debugMode = False

#Should progress bars be visualized with ascii characters (default: False)
asciiFlag = False

#Method overwrite file; will execute specified file to overwrite otherwise defined methods/parameters (default: None)
overWriteFile = None

#==================================================================
#L4-1: CLASSIFICATION
#==================================================================

#Define labels used for normal/benign tissue
#'a': normal adipose.
#'s': normal stroma tissue excluding adipose.
#'o': other normal tissue including parenchyma, adenosis, lobules, blood vessels, etc.
labelsBenign = ['a', 's', 'o', 'normal']

#Define labels used for malignant tissue
#'d': IDC tumor
#'l': ILC tumor
#'ot': other tumor areas including DCIS, biopsy site, and slightly defocused tumor regions.
labelsMalignant = ['d', 'l', 'ot', 'tumor']

#Define labels used for tissues to be excluded
#'ft': defocused but still visually tumor-like areas.
#'f': severly out-of-focusing areas. 
#'b': background. 
#'e': bubbles.
labelsExclude = ['ft', 'f', 'b', 'e', 'exclude']

#==================================================================

#==================================================================
#L4-2: RECONSTRUCTION
#==================================================================
#TBD
#==================================================================

#==================================================================
#L4-3: SIMULATION
#==================================================================

#Define precision of the percentage averaging (as percentage is inconsistent between acquistion steps) (default: 0.001)
precision = 0.001

#If performing a benchmark, should processing be skipped (default: False)
benchmarkNoProcessing = False

#Should simulated sampling in testing be bypassed to work on OOM errors; relevant sections must have already been run successfully (default: False)
bypassSampling = False

#Should result objects generated in testing be kept (only needed if bypassSampling is to be used later) (default: False)
keepResultData = False

#==================================================================

###########################################################################################################



###########################################################################################################
#L5: ISOLATED PARAMETERS - DO NOT CHANGE, ALTERNATIVE OPTIONS NOT CURRENTLY FUNCTIONAL
###########################################################################################################
#TBD
###########################################################################################################



###########################################################################################################
#L6: DEBUG/DEPRECATED - LIKELY TO BE REMOVED IN FUTURE
###########################################################################################################
#TBD
###########################################################################################################


