#====================================================================
#CONFIGURATION
#Adding definitions here can be used to overide defaults in CODE/CONFIGURATION.py
#====================================================================

#Should classification/cross-validation be performed on patches
classifierTrain = True

#Should classifier model components be saved/exported
classifierExport = True

#Should classifier model components be evaluated
classifierEvaluate = True

#What mechansim should be used for WSI evaluation: 'gradcam++', 'majority'
evaluateMethodWSI = 'gradcam++'

#Should available synthetic patches be used in training
addSyntheticPatches = False

#At what ratio of malignant to benign patches should a WSI be predicted to be malignant 
thresholdWSI_prediction = 0.15

#Specify folds for cross validation if desired (e.g. [['1', '3'], ['4', '2']]), else specify number of folds to generate
manualFolds = [['2_1', '9_3', '11_3', '16_3', '34_1', '36_2', '40_2', '54_2', '57_2', '60_1', '62_1'],
               ['17_5', '20_3', '23_3', '24_2', '28_2', '30_2', '33_3', '51_2', '52_2', '59_2', '63_3', '66_2'], 
               ['12_1', '14_2', '22_3', '26_3', '35_4', '44_1', '45_1', '47_2', '49_1', '53_2', '56_2', '68_1'], 
               ['4_4', '5_3', '8_1', '10_3', '25_3', '27_1', '29_2', '37_1', '42_3', '48_3', '50_1', '69_1'], 
               ['7_2', '15_4', '19_2', '31_1', '43_1', '46_2', '55_2', '58_2', '61_1', '64_1', '65_1', '67_1', '70_1']]
