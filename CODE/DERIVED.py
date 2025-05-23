#==================================================================
#DERIVED
#==================================================================

#Define general labels and values to use
labelBenign, labelMalignant, labelExclude = '0', '1', '2'
valueBenign, valueMalignant, valueBackground = int(labelBenign), int(labelMalignant), 2
cmapClasses = colors.ListedColormap(['lime', 'red', 'black'])

#Determine offset to avoid overlapping squares in grid visualization
gridThicknessOffset = gridThickness//2