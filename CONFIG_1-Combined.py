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
addSyntheticPatches = True

#At what ratio of malignant to benign patches should a WSI be predicted to be malignant 
thresholdWSI_prediction = 0.2

#Specify folds for cross validation if desired (e.g. [['1', '3'], [S4', '2']]), else specify number of folds to generate
manualFolds = [['131_1', '103_1', '101_1', '44_1', '17_5', '55_2', '143_1', '169_1', '139_1', '172_1', '62_1', '154_1', '52_2', '83_1', '114_2', '32_2', '30_2', '160_1', '130_1', '65_1', '70_1', '95_1', '146_1', '92_1', '85_1', '11_3', '112_2', '45_1', '128_2', '125_1', '14_2'],
               ['51_2', '110_2', '67_1', '20_3', '21_1', '79_1', '173_1', '22_3', '54_2', '31_1', '81_1', '26_3', '36_2', '5_3', '61_1', '129_1', '29_2', '96_1', '24_2', '142_1', '53_2', '100_1', '35_4', '16_3', '37_1', '86_1', '34_1', '120_1', '10_3'],
               ['3_1', '149_1', '58_2', '78_1', '134_2', '59_2', '28_2', '90_1', '165_1', '75_1', '68_1', '74_1', '145_1', '99_1', '40_2', '8_1', '135_1', '105_1', '104_1', '97_2', '76_1', '137_2', '82_1', '6_1', '56_2', '136_1', '15_4', '23_3'],
               ['73_1', '98_1', '66_2', '132_2', '7_2', '64_1', '166_1', '38_1', '122_2', '88_1', '43_1', '4_4', '49_1', '63_3', '138_1', '50_1', '57_2', '119_1', '106_1', '2_1', '93_1', '48_3', '94_1', '123_1', '25_3', '69_1', '19_2'],
               ['126_2', '47_2', '133_1', '144_1', '27_1', '174_1', '46_2', '179_1', '109_1', '113_1', '91_1', '141_1', '13_1', '9_3', '127_1', '84_1', '12_1', '72_1', '118_1', '111_1', '80_1', '60_1', '124_1', '140_1', '77_1', '42_3', '33_3']]
