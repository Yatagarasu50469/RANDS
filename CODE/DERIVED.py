#==================================================================
#DERIVED
#==================================================================

#Compute the ratio of minimum foreground to patch area required for a patch to be considered as holding foreground data
patchBackgroundRatio = minimumForegroundArea/((patchSize*cameraResolution)**2)
if patchBackgroundRatio > 1.0: sys.exit('\nError - Minimum foreground area specified for foreground data exceeds the given patch size.\n')

#Define general labels and values to use
labelBenign, labelMalignant, labelExclude = '0', '1', '2'
valueBenign, valueMalignant, valueBackground = int(labelBenign), int(labelMalignant), 2
cmapClasses = colors.ListedColormap(['lime', 'red', 'black'])

#Determine offset to avoid overlapping squares in grid visualization
gridThicknessOffset = gridThickness//2

