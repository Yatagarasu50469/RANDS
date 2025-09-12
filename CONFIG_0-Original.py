#====================================================================
#CONFIGURATION - Overrides defaults in CODE/CONFIGURATION.py
#====================================================================

#Should classification/cross-validation be performed on patches
classifierTrain = True

#Should classifier model components be saved/exported
classifierExport = True

#Should classifier model components be evaluated
classifierEvaluate = True

#Which channel should be used to assess whether patches are in the foreground: 'red', 'green', 'gray'
channel_extraction = 'gray'

#Ratio of chosen-channel (channel_extraction) values >= backgroundLevel in a patch, below which the patch will not be considered
thresholdPatch_extraction = 0.8

#What ratio of malignant to benign patches should be used to predict a whole WSI as malignant 
thresholdWSI_prediction = 0.15
