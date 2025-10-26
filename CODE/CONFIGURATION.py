#==================================================================
#CONFIGURATION - Default parameters overridden by CONFIG files
#==================================================================

##################################################################
#L0: TASKS TO BE PERFORMED
##################################################################

#Should classification/cross-validation be performed on patches
classifierTrain = False

#Should classifier model components be saved/exported
classifierExport = False

#Should classifier model components be evaluated
classifierEvaluate = False

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

#How many samples should be submitted in a batch through pytorch models used in classifier (default: 1)
#WARNING: Values > 1 affect the order of operations and computation optimizations, introducing unpredicable artifacts that do change the results!
#Ref: https://discuss.pytorch.org/t/varying-batch-size-for-pre-trained-model-changes-inference-result-even-in-evaluation-mode/223688
#At max, ResNet50 & DenseNet169 with a 2080TI 11GB can handle 64 with resizeSize=224 or 16 with resizeSize=0
batchsizeClassifier = 1

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

#Which classifier model file should be loaded: 'xgb' or 'vit' (default: 'xgb')
classifierModel = 'xgb'

#What mechansim should be used for WSI evaluation: 'gradcam++', 'majority' (default: 'gradcam++')
evaluateMethodWSI = 'gradcam++'

#Should available synthetic patches be used in training (default: True)
addSyntheticPatches = True

#At what ratio of malignant to benign patches should a WSI be predicted to be malignant  (default: 0.15)
#When set to 0, will label a WSI as malignant if any one patch is predicted as malignant
#Unknown what the original work used for this value, but for original work replication, a value of 0.15 seems appropriate 
#Replication of original results occurs with values between 0.12-0.19
#If this value is changed, then should enable visualizeLabelGrids_patches to update stored data
thresholdWSI_prediction = 0.15

#Which channel should be used to assess whether patches are in the foreground: 'red', 'green', 'gray' (default: 'red')
#'gray' used for original work replication
channel_extraction = 'red'

#At what ratio of chosen-channel (channel_extraction) values >= backgroundLevel in a patch should a patch be considered (default: 0.2)
#0.8 used for original work replication
thresholdPatch_extraction = 0.2

#Minimum chosen channel value [0, 255] for a location to contribute to the backgroundThreshold criteria (default: 5)
backgroundLevel = 5

#When splitting WSI images, what size should the resulting patches be (default: 400)
#Should remain consistent with patch sizes given for training
patchSize = 400

#Specify what symmetrical dimension patches should be resized to; if no resizing is desired leave as 0 (default: 224)
#Leaving as 0 will increase training time (also may need to lower batchsizeClassifier), but can lead to improved scores
#Original implementation uses 224, though with adaptive average pooling this isn't actually neccessary
resizeSize_patches = 224

#Specify what symmetrical dimension WSI should be resized to when generating saliency maps (default: 224)
#If fusion method were to be further developed, this should be changed to maintain the original sample aspect ratio. 
resizeSize_WSI = 224

#==================================================================
#L2-1: CROSS-VALIDATION
#==================================================================

#Specify folds for cross validation if desired (e.g. [['1', '3'], ['4', '2']]), else specify number of folds to generate (default: 5)
manualFolds = 5

#Dataset 1
##############################################################################################################################
#Default matches folds used in prior work (https://doi.org/10.3389/fonc.2023.1179025)
#Omits 6 available samples, with folds holding: 11, 12, 12, 12, 13 samples respectively; this may have been to better balance class distribution
#Presently, all available (non-excluded) samples (not just those in manualFolds) are currently used for training the exported/utilized final classifier
#manualFolds = [['2_1', '9_3', '11_3', '16_3', '34_1', '36_2', '40_2', '54_2', '57_2', '60_1', '62_1'],
#               ['17_5', '20_3', '23_3', '24_2', '28_2', '30_2', '33_3', '51_2', '52_2', '59_2', '63_3', '66_2'], 
#               ['12_1', '14_2', '22_3', '26_3', '35_4', '44_1', '45_1', '47_2', '49_1', '53_2', '56_2', '68_1'], 
#               ['4_4', '5_3', '8_1', '10_3', '25_3', '27_1', '29_2', '37_1', '42_3', '48_3', '50_1', '69_1'], 
#               ['7_2', '15_4', '19_2', '31_1', '43_1', '46_2', '55_2', '58_2', '61_1', '64_1', '65_1', '67_1', '70_1']]

#Datset 2
##############################################################################################################################
#Generate randomized and approximately class-balanced set
#classLabels = np.asarray(np.zeros(len(samplesBenign)).astype(int).tolist() + np.ones(len(samplesMalignant)).astype(int).tolist() + (np.ones(len(samplesMixed)).astype(int)*2).tolist())
#samplesBenign = ['72_1', '74_1', '80_1', '83_1', '96_1', '98_1', '100_1', '118_1', '119_1', '120_1', '122_2', '123_1',
#                '124_1', '126_2', '127_1', '128_2', '129_1', '130_1', '131_1', '134_2', '137_2', '138_1', '140_1', '141_1',
#                '143_1', '144_1', '145_1', '146_1', '149_1', '154_1', '165_1', '169_1', '172_1', '173_1']
#samplesMalignant = ['75_1', '84_1', '88_1', '91_1', '99_1', '101_1', '105_1']
#sampleNames = natsort.natsorted([os.path.splitext(name)[0] for name in os.listdir(dir_patches_inputWSI)])
#samplesMixed = natsort.natsorted([sampleName for sampleName in [sampleName for sampleName in sampleNames if sampleName not in samplesBenign] if sampleName not in samplesMalignant])
#samplesTotal = np.asarray(samplesBenign + samplesMalignant + samplesMixed)
#np.random.seed(0)
#manualFolds, manualFoldsLabels = [], []
#for trainSamples, testSamples in sklearn.model_selection.StratifiedKFold(n_splits=5, shuffle=True).split(samplesTotal, classLabels): 
#    manualFolds.append(samplesTotal[testSamples].tolist())
#    manualFoldsLabels.append(classLabels[testSamples].tolist())
#Remove 96_1 due to visible issues resulting from correction process
#manualFolds = [[ '129_1', '137_2', '140_1', '143_1', '172_1', '173_1', '75_1', '84_1', '106_1', '109_1', '110_2', '113_1', '133_1', '142_1', '174_1'], 
#               ['72_1', '119_1', '126_2', '127_1', '138_1', '149_1', '154_1', '88_1', '77_1', '82_1', '85_1', '104_1', '132_2', '139_1', '166_1'], 
#               ['74_1', '83_1', '118_1', '124_1', '130_1', '144_1', '146_1', '105_1', '81_1', '86_1', '90_1', '95_1', '103_1', '125_1', '136_1'], 
#               ['80_1', '98_1', '123_1', '134_2', '141_1', '145_1', '169_1', '91_1', '73_1', '78_1', '79_1', '111_1', '114_2', '135_1', '160_1'], 
#               ['100_1', '120_1', '122_2', '128_2', '131_1', '165_1', '99_1', '101_1', '76_1', '92_1', '93_1', '94_1', '97_2', '112_2', '179_1']]

#Label distribution (0: benign; 1: malignant; 2: mixed)
#[0, 0, 0, 0, 0, 0, 1, 1, 2, 2, 2, 2, 2, 2, 2]    #Total: 15; Benign: 7; Malignant: 2; Mixed: 7
#[0, 0, 0, 0, 0, 0, 0, 1, 2, 2, 2, 2, 2, 2, 2]    #Total: 15; Benign: 7; Malignant: 1; Mixed: 7
#[0, 0, 0, 0, 0, 0, 0, 1, 2, 2, 2, 2, 2, 2, 2]    #Total: 15; Benign: 7; Malignant: 1; Mixed: 7
#[0, 0, 0, 0, 0, 0, 0, 1, 2, 2, 2, 2, 2, 2, 2]    #Total: 15; Benign: 7; Malignant: 1; Mixed: 7
#[0, 0, 0, 0, 0, 0, 1, 1, 2, 2, 2, 2, 2, 2, 2]    #Total: 15; Benign: 6; Malignant: 2; Mixed: 7

#============================================================================================================================

#==================================================================
#L2-2: CLASSIFICATION MODELS
#==================================================================

#******************************************************************
#L2-2-1: ORIGINAL
#******************************************************************

#How many processes should be used for preprocessing patch data; set to -1 for maximum possible
#Decrease if OOM occurs during feature extraction
numWorkers_patches = -1

#How many processes should be used for preprocesisng WSI data; set to -1 for maximum possible
#Decrease if OOM occurs during saliency mapping
numWorkers_WSI = -1

#Should features be extracted for patches and overwrite previously generated files
#Patch features do not have associated patch names; if samples used are changed, features must be recomputed!
overwrite_patches_features = True

#Should saliency maps be determined for patches and overwrite previously generated files
#Patch weights do not have associated patch names; if samples used are changed, patch weights and originating saliency maps must be recomputed!
overwrite_patches_saliencyMaps = True

#Should saliency maps and their overlays be visualized for patch WSI
visualizeSaliencyMaps_patches = True

#Should label grids and their overlays be visualized for patch WSI; will overwrite previously generated files
#Files should be updated if thresholdWSI is changed
visualizeLabelGrids_patches = True

#Should prediction grids and their overlays be visualized for patch WSI
visualizePredictionGrids_patches = True

#Which pre-trained weight sets should be used for ResNet and DenseNet model: 'IMAGENET1K_V1', 'IMAGENET1K_V2'
#Unclear which weights were used for TensorFlow ResNet50 variant in original implementation, but V2 did improve scores a bit when using resizeSize = 0
weightsResNet = 'IMAGENET1K_V2'

#Which pre-trained weight sets should be used for DenseNet model: 'IMAGENET1K_V1'
weightsDenseNet = 'IMAGENET1K_V1'

#******************************************************************
#L2-2-2: UPDATED
#******************************************************************
#TBD
#******************************************************************

#==================================================================

#==================================================================
#L2-3: RECONSTRUCTION DATA GENERATION
#==================================================================

#Should WSI preparation and patch extraction (for patch-specific WSI) overwrite previously generated files
overwrite_recon_patches = True

#Should features be extracted for WSI extracted patches (for patch-specific WSI) and overwrite previously generated files
overwrite_recon_features = True

#Should saliency maps be determined for WSI extracted patches (for patch-specific WSI) and overwrite previously generated files
overwrite_recon_saliencyMaps = True

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

#How thick should the grid lines be when generating overlay images (defualt: 50)
gridThickness = 50

#Should saliency map overlays be placed over data converted to grayscale for clearer visualization (default: True)
overlayGray = True

#Should images be saved to lossless (.tif) or compressed (.jpg) image format (default: False)
#Any images that are anticipated to be reused (such as extracted patches), should be hardcoded to be saved in a lossless format by default
exportLossless = False

#For .jpg image outputs, what should the compression quality (%) be (default: 90)
#WARNING: Setting to 100 is not sufficient to generate in lossless/exact outputs; if that is desired, use overlayLossless instead! 
exportQuality = 90

#What weight should be used when overlaying data
overlayWeight = 0.5

#RNG seed value to help ensure run-to-run consistency; set to -1 to disable
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
#'N': normal
labelsBenign = ['a', 's', 'o', 'normal', 'N']

#Define labels used for malignant tissue
#'d': IDC tumor
#'l': ILC tumor
#'ot': other tumor areas including DCIS, biopsy site, and slightly defocused tumor regions.
#'T': tumor
labelsMalignant = ['d', 'l', 'ot', 'tumor', 'T']

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


