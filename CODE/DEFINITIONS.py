#==================================================================
#DEFINITIONS
#==================================================================

#Load/synchronize data labeling, drop excluded rows, and extract relevant metadata
def loadMetadata_patches(filename):
    metadata = pd.read_csv(filename, header=0, names=['Sample', 'Index', 'Row', 'Column', 'Label'], converters={'Sample':str,'Index':str, 'Row':int, 'Column':int, 'Label':str})
    metadata['Label'] = metadata['Label'].replace(labelsBenign, labelBenign)
    metadata['Label'] = metadata['Label'].replace(labelsMalignant, labelMalignant)
    metadata['Label'] = metadata['Label'].replace(labelsExclude, labelExclude)
    metadata = metadata.loc[metadata['Label'] != labelExclude]
    return [np.squeeze(data) for data in np.split(np.asarray(metadata), [1, 2, 4], -1)]

#Extract patches and determine associated metadata for referenced WSI files; abstraction allows for isolation of WSI data used for training the classifier
def extractPatches(WSIFilenames, patchBackgroundValue, dir_output):
    
    #Extract uniform, non-overlapping patches from each WSI; may be too memory intensive and RW-bottlenecked to parallelize efficiently
    patchNames, patchFilenames, patchSampleNames, patchLocations, cropData, paddingData, shapeData = [], [], [], [], [], [], []
    for filename in tqdm(WSIFilenames, desc='Patch Extraction', leave=True, ascii=asciiFlag):
        
        #Extract base sample name
        sampleName = os.path.basename(filename).split('.jpg')[0]
        
        #Load WSI image
        imageWSI = cv2.cvtColor(cv2.imread(filename, cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB)
        
        #Tried a few quicker methods for pre-processing before applying some foreground segmentation
        #No tangible benefits were observed in performance, but leaving in code at least once for archival reference
        
        #Histogram equalization
        #y, cr, cb = cv2.split(cv2.cvtColor(imageWSI, cv2.COLOR_RGB2YCrCb))
        #imageMask = cv2.cvtColor(cv2.merge((cv2.equalizeHist(y), cr, cb)), cv2.COLOR_YCR_CB2RGB)
        
        #Denoising
        #imageMask = cv2.fastNlMeansDenoising(cv2.cvtColor(imageWSI, cv2.COLOR_RGB2GRAY), None, h=3, templateWindowSize=7, searchWindowSize=21)
        
        #CLAHE
        #imageMask = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8)).apply(cv2.cvtColor(imageWSI, cv2.COLOR_RGB2GRAY))
        
        #Isolate only the largest non-zero area; would not be able to handle samples with disconnected tissue segments
        #imageMask = cv2.threshold(cv2.cvtColor(imageWSI, cv2.COLOR_RGB2GRAY),0,255,cv2.THRESH_BINARY)[1]
        #imageMask = max(cv2.findContours(imageMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0], key=cv2.contourArea)
        
        #Crop WSI to the foreground area using Otsu
        imageWSI_gray = cv2.cvtColor(imageWSI, cv2.COLOR_RGB2GRAY)
        x, y, w, h = cv2.boundingRect(cv2.threshold(imageWSI_gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]) 
        imageWSI = imageWSI[y:y+h, x:x+w]
        imageWSI_gray = imageWSI_gray[y:y+h, x:x+w]
        cropData.append([y, y+h, x, x+w])
        
        #Pad the image as needed (as symmetrially as possible) for an even division by the specified patch size; compute numPatches per row/column
        padHeight = (int(np.ceil(imageWSI.shape[0]/patchSize))*patchSize)-imageWSI.shape[0]
        padWidth = (int(np.ceil(imageWSI.shape[1]/patchSize))*patchSize)-imageWSI.shape[1]
        padTop, padLeft = padHeight//2, padWidth//2
        padBottom, padRight = padTop+(padHeight%2), padLeft+(padWidth%2)
        imageWSI = np.pad(imageWSI, ((padTop, padBottom), (padLeft, padRight), (0, 0)))
        imageWSI_gray = np.pad(imageWSI_gray, ((padTop, padBottom), (padLeft, padRight)))
        paddingData.append([padTop, padBottom, padLeft, padRight])
        numPatchesRow, numPatchesCol = imageWSI.shape[0]//patchSize, imageWSI.shape[1]//patchSize
        shapeData.append([numPatchesRow, numPatchesCol])
        
        #Split the WSI (color and grayscale) into patches and flatten
        imageWSI = imageWSI.reshape(numPatchesRow, patchSize, numPatchesCol, patchSize, imageWSI.shape[2]).swapaxes(1,2)
        imageWSI = imageWSI.reshape(-1, imageWSI.shape[2], imageWSI.shape[3], imageWSI.shape[4])
        imageWSI_gray = imageWSI_gray.reshape(numPatchesRow, patchSize, numPatchesCol, patchSize).swapaxes(1,2)
        imageWSI_gray = imageWSI_gray.reshape(-1, imageWSI_gray.shape[2], imageWSI_gray.shape[3])
        
        #Setup directory to store patches
        dir_patches = dir_output + 'S' + sampleName + os.path.sep
        if not os.path.exists(dir_patches): os.makedirs(dir_patches)
        
        #Record metadata for each patch that has a specified percentage of foreground data and save each to disk (always lossless)
        patchIndex = 0
        for rowNum in range(0, numPatchesRow):
            for colNum in range(0, numPatchesCol):
                if np.mean(imageWSI_gray[patchIndex] >= patchBackgroundValue) >= patchBackgroundRatio: 
                    locationRow, locationColumn= rowNum*patchSize, colNum*patchSize
                    patchLocations.append([locationRow, locationColumn])
                    patchName = sampleName+'_'+str(patchIndex)+'_'+str(locationRow)+'_'+str(locationColumn)
                    patchNames.append(patchName)
                    patchSampleNames.append(sampleName)
                    patchFilenames.append(exportImage(dir_patches+'PS'+patchName, imageWSI[patchIndex], True))
                patchIndex += 1
    
    return np.asarray(patchNames), np.asarray(patchFilenames), np.asarray(patchSampleNames), np.asarray(patchLocations), np.asarray(cropData), np.asarray(paddingData), np.asarray(shapeData)

#Compute metrics for a classification result and visualize/save them as needed
def computeClassificationMetrics(labels, predictions, baseFilename, suffix):
    
    #Specify data class labels
    displayLabels = ['Benign', 'Malignant']
    classLabels = np.arange(0, len(displayLabels), 1)
    
    #Generate/store a classification report: precision, recall, f1-score, and support
    classificationReport = classification_report(labels, predictions, labels=classLabels, target_names=displayLabels, output_dict=True)#, zero_division=0.0)
    classificationReport = pd.DataFrame(classificationReport).transpose()
    classificationReport.to_csv(baseFilename + 'classReport' + suffix + '.csv')

    #Generate a confusion matrix and extract relevant statistics
    confusionMatrix = confusion_matrix(labels, predictions, labels=classLabels)
    tn, fp, fn, tp = confusionMatrix.ravel()
    accuracy = (tp+tn)/(tp+tn+fp+fn)
    sensitivity = tp/(tp+fn)
    specificity = tn/(tn+fp)
    
    #Store relevant statistics
    summaryReport = {'Accuracy': accuracy, 'Sensitivity': sensitivity, 'Specificity': specificity}
    summaryReport = pd.DataFrame.from_dict(summaryReport, orient='index')
    summaryReport.to_csv(baseFilename + 'summaryReport' + suffix + '.csv')
    
    #Store confusion matrix
    displayCM = ConfusionMatrixDisplay(confusionMatrix, display_labels=displayLabels)
    displayCM.plot(cmap='Blues')
    plt.tight_layout()
    if exportLossless: plt.savefig(baseFilename+'confusionMatrix' + suffix + '.tif')
    else: plt.savefig(baseFilename+'confusionMatrix' + suffix + '.jpg')
    plt.close()
    
#Convert numpy array to contiguous tensor; issues with lambda functions when using multiprocessing
def contiguousTensor(inputs):
    return torch.from_numpy(inputs).contiguous()
    
#Rescale tensor; issues with lambda functions when using multiprocessing
def rescaleTensor(inputs):
    return inputs.to(dtype=torch.get_default_dtype()).div(255)
    
#Generate a torch transform needed for preprocessing image data
def generateTransform(resizeSize=[], rescale=False, normalize=False):
    transform = [contiguousTensor]
    if len(resizeSize) > 0: transform.append(v2.Resize(tuple(resizeSize)))
    if rescale: transform.append(rescaleTensor)
    if normalize: transform.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    return transforms.Compose(transform)

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