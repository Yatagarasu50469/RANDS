#==================================================================
#DERIVED
#==================================================================

#Compute the ratio of minimum foreground to block area required for a block to be considered as holding foreground data
blockBackgroundRatio = minimumForegroundArea/((blockSize*cameraResolution)**2)
if blockBackgroundRatio > 1.0: sys.exit('\nError - Minimum foreground area specified for foreground data exceeds the given block size.')

#Define general labels and values to use
labelBenign, labelMalignant, labelExclude = '0', '1', '2'
valueBenign, valueMalignant, valueBackground = int(labelBenign), int(labelMalignant), 2
cmapClasses = colors.ListedColormap(['lime', 'red', 'black'])

#Determine overlay and grid image export extension; exporting as .tif for losselss and .jpg for compressed
if overlayLossless: overlayExtension = '.tif'
else: overlayExtension = '.jpg'

#Determine offset to avoid overlapping squares in grid visualization
gridThicknessOffset = gridThickness//2

