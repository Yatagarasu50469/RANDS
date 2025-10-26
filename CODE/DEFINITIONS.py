#==================================================================
#DEFINITIONS
#==================================================================

#Load/synchronize data labeling, drop excluded rows, and extract relevant metadata
def loadMetadata_patches(filename):
    
    #Load patch-level metadata; may or may not contain Edge/Boundary columns
    try: metadata = pd.read_csv(filename, header=0, names=['Sample', 'Index', 'Row', 'Column', 'Label', 'Edge', 'Boundary'], converters={'Sample':str,'Index':str, 'Row':int, 'Column':int, 'Label':str, 'Edge':str, 'Boundary':str})
    except: metadata = pd.read_csv(filename, header=0, names=['Sample', 'Index', 'Row', 'Column', 'Label'], converters={'Sample':str,'Index':str, 'Row':int, 'Column':int, 'Label':str})
    
    #Sometimes patch filenames were used instead of sample name in metadata
    patchSampleNames = metadata['Sample'].to_numpy().tolist()
    if len(patchSampleNames[0].split('_')) > 2: 
        patchSampleNames = ['_'.join(patchSampleName.split('_')[:2]) for patchSampleName in patchSampleNames]
        metadata['Sample'] = patchSampleNames
        
    #Not currently using edge or boundary labels; may not be present
    try: 
        del metadata['Edge']
        del metadata['Boundary']
    except: 
        pass
    
    #Unify benign/malignant/exclusion labels
    metadata['Label'] = metadata['Label'].replace(labelsBenign, labelBenign)
    metadata['Label'] = metadata['Label'].replace(labelsMalignant, labelMalignant)
    metadata['Label'] = metadata['Label'].replace(labelsExclude, labelExclude)
    metadata = metadata.loc[metadata['Label'] != labelExclude]
    return [np.squeeze(data) for data in np.split(np.asarray(metadata), [1, 2, 4], -1)]

#Load/synchronize data labeling, drop excluded rows, and extract relevant metadata
def loadMetadata_WSI(filename):
    metadata = pd.read_csv(filename, header=0, names=['Sample', 'Label'], converters={'Sample':str, 'Label':str})
    metadata['Label'] = metadata['Label'].replace(labelsBenign, labelBenign)
    metadata['Label'] = metadata['Label'].replace(labelsMalignant, labelMalignant)
    metadata['Label'] = metadata['Label'].replace(labelsExclude, labelExclude)
    metadata = metadata.loc[metadata['Label'] != labelExclude]
    return [np.squeeze(data) for data in np.split(np.asarray(metadata), 2, -1)]

def extractPatches(imageWSI, dir_patches):

    #Crop off the right and bottom edges for even division by the specified patch size
    numPatchesRow, numPatchesCol = imageWSI.shape[0]//patchSize, imageWSI.shape[1]//patchSize
    y, h, x, w = 0, numPatchesRow*patchSize, 0, numPatchesCol*patchSize
    imageWSI = imageWSI[y:y+h, x:x+w]
    
    #Store relevant parameters for later use
    cropData = [y, y+h, x, x+w]
    paddingData = [None, None, None, None]
    numPatchesRow, numPatchesCol = imageWSI.shape[0]//patchSize, imageWSI.shape[1]//patchSize
    shapeData = [numPatchesRow, numPatchesCol]
    
    #Split the WSI into patches and flatten
    imageWSI = imageWSI.reshape(numPatchesRow, patchSize, numPatchesCol, patchSize, imageWSI.shape[2]).swapaxes(1,2)
    imageWSI = imageWSI.reshape(-1, imageWSI.shape[2], imageWSI.shape[3], imageWSI.shape[4])
    
    #Save patches to disk (always lossless) that meet a given threshold of chosen-channel values at or over a given background level
    patchLocations, patchNames, patchSampleNames, patchFilenames = [], [], [], []
    patchIndex = 0
    for rowNum in range(0, numPatchesRow):
        for colNum in range(0, numPatchesCol):
            image = imageWSI[patchIndex]
            if channel_extraction == 'red': channelValue = np.mean(image[:,:,0] >= backgroundLevel)
            elif channel_extraction == 'green': channelValue = np.mean(image[:,:,1] >= backgroundLevel)
            elif channel_extraction == 'gray': channelValue = np.mean(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) >= backgroundLevel)
            else: sys.exit('Error - Unknown channel selected for use during patch extraction')
            if channelValue > thresholdPatch_extraction:
                locationRow, locationColumn= rowNum*patchSize, colNum*patchSize
                patchLocations.append([locationRow, locationColumn])
                patchName = sampleName+'_'+str(patchIndex)+'_'+str(locationRow)+'_'+str(locationColumn)
                patchNames.append(patchName)
                patchSampleNames.append(sampleName)
                patchFilename = dir_patches + 'PS' + patchName + '.tif'
                writeSuccess = cv2.imwrite(patchFilename, cv2.cvtColor(image, cv2.COLOR_RGB2BGR), params=(cv2.IMWRITE_TIFF_COMPRESSION, 1))
                patchFilenames.append(patchFilename)
            patchIndex += 1

    return cropData, shapeData, paddingData, patchLocations, patchNames, patchSampleNames, patchFilenames

#Extract patches and determine associated metadata for referenced WSI files; abstraction allows for isolation of WSI data used for training the classifier
def extractPatchesMultiple(WSIFilenames, patchBackgroundValue, dir_output):
    
    #Extract uniform, non-overlapping patches from each WSI; may be too memory intensive and RW-bottlenecked to parallelize efficiently
    patchNamesTotal, patchFilenamesTotal, patchSampleNamesTotal, patchLocationsTotal, cropDataTotal, paddingData, shapeData = [], [], [], [], [], [], []
    for filename in tqdm(WSIFilenames, desc='Patch Extraction', leave=True, ascii=asciiFlag):
        
        #Extract base sample name
        sampleName = os.path.basename(filename).split('.jpg')[0]
        
        #Load the WSI image
        imageWSI = cv2.cvtColor(cv2.imread(filename, cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB)
        
        #Setup a directory to store patches
        dir_patches = dir_output + 'S' + sampleName + os.path.sep
        if not os.path.exists(dir_patches): os.makedirs(dir_patches)
        
        #Extract patches from the WSI image
        cropData, shapeData, paddingData, patchLocations, patchNames, patchSampleNames, patchFilenames = extractPatches(imageWSI, dir_patches)
        cropDataTotal += cropData
        shapeDataTotal += shapeData
        paddingDataTotal += paddingData
        patchLocationsTotal += patchLocations
        patchNamesTotal += patchNames
        patchSampleNamesTotal += patchSampleNames
        patchFilenamesTotal += patchFilenames
        
    return np.asarray(patchNamesTotal), np.asarray(patchFilenamesTotal), np.asarray(patchSampleNamesTotal), np.asarray(patchLocationsTotal), np.asarray(cropDataTotal), np.asarray(paddingDataTotal), np.asarray(shapeDataTotal)

#Compute metrics, exporting resulting confusion matrix
def computeClassificationMetrics(labels, predictions):
    confusionMatrix = confusion_matrix(labels, predictions, labels=[valueBenign, valueMalignant])
    tn, fp, fn, tp = confusionMatrix.ravel()
    statistics = {}
    statistics['Accuracy'] = (tp+tn)/(tp+tn+fp+fn)
    statistics['Sensitivity/Recall'] = tp/(tp+fn)
    statistics['Specificity'] = tn/(tn+fp)
    statistics['Precision'] = tp/(tp+fp)
    statistics['f1-Score'] = (2*tp)/(2*tp+fp+fn)
    return statistics
    
#Export a confusion matrix visualization
def exportConfusionMatrix(labels, predictions, baseFilename, suffix): 
    confusionMatrix = confusion_matrix(labels, predictions, labels=[valueBenign, valueMalignant])
    displayCM = ConfusionMatrixDisplay(confusionMatrix, display_labels=['Benign', 'Malignant'])
    displayCM.plot(cmap='Blues')
    plt.tight_layout()
    if exportLossless: plt.savefig(baseFilename+'confusionMatrix' + suffix + '.tif')
    else: plt.savefig(baseFilename+'confusionMatrix' + suffix + '.jpg')
    plt.close()

#Consolidate per-fold statistics into a single dataframe, computing average and standard deviation if needed
def consolidateStatistics(globalStatistics, numFolds): 
    finalStatistics = {metric: [statistics[metric] for statistics in globalStatistics] for metric in globalStatistics[0]}
    if numFolds > 1:
        for metric in globalStatistics[0]: finalStatistics[metric] += [np.mean(finalStatistics[metric]), np.std(finalStatistics[metric])]
        foldValues = {'Fold': np.arange(1, numFolds+1, dtype=int).astype(str).tolist()+['Avg', 'Std']}
        finalStatistics = {**foldValues, **finalStatistics}
    finalStatistics = pd.DataFrame.from_dict(finalStatistics, orient='index')
    if numFolds == 1: finalStatistics = finalStatistics.drop(['Fold'])
    return finalStatistics

#Export lossless RGB image data to disk
def exportImage(filename, image, exportLosslessFlag):
    if exportLosslessFlag: 
        filename += '.tif'
        writeSuccess = cv2.imwrite(filename, cv2.cvtColor(image, cv2.COLOR_RGB2BGR), params=(cv2.IMWRITE_TIFF_COMPRESSION, 1))
    else: 
        filename += '.jpg'
        writeSuccess = cv2.imwrite(filename, cv2.cvtColor(image, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), exportQuality])
    if not writeSuccess: sys.exit('\nError - Unable to write file to disk; please verify directory location exists and has sufficient free space: ' + filename + '\n')
    return filename

#OpenCV does not output sharp corners with its rectangle method...unless it's filled in
def rectangle(image, startPos, endPos, color):
    image = cv2.rectangle(image, startPos, endPos, color, -1)
    image = cv2.rectangle(image, (startPos[0]+gridThicknessOffset, startPos[1]+gridThicknessOffset), (endPos[0]-gridThicknessOffset, endPos[1]-gridThicknessOffset), (0, 0, 0), -1)
    return image
    
#Define how to reset the random seed for deterministic repeatable RNG
def resetRandom():
    if manualSeedValue != -1:
        torch.manual_seed(manualSeedValue)
        torch.cuda.manual_seed_all(manualSeedValue)
        np.random.seed(manualSeedValue)
        random.seed(manualSeedValue)