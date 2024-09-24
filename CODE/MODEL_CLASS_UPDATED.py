#==================================================================
#MODEL_CLASS_ORIGINAL
#==================================================================

#Load and preprocess data image files; avoid storing data in VRAM
#ToTensor() swaps axes (so not used); convert to contiguous torch tensor manually
#Rescaling and changing data type only after resizing (otherwise can escape [0, 1])
class DataPreprocessing_Classifier(Dataset):
    def __init__(self, filenames, resizeSize):
        super().__init__()
        self.filenames = filenames
        self.numFiles = len(self.filenames)
        if resizeSize > 0: self.transform = generateTransform([resizeSize, resizeSize], True, True)
        else: self.transform = generateTransform([], True, True)
        
    #Rearranging import dimensions, allows resize transform before tensor-conversion/rescaling; preserves precision and output data range
    def __getitem__(self, index): return self.transform(np.moveaxis(cv2.cvtColor(cv2.imread(self.filenames[index], cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB), -1, 0))
    
    def __len__(self): return self.numFiles
    
#Classify image files
class Classifier():
    
    #Load blocks specified by provided filenames, extract relevant features, and setup additional model components needed for training/evaluation
    def __init__(self, dataType, blockNames, blockFilenames, blockSampleNames, blockLocations, sampleNames, WSIFilenames, WSILabels, cropData=[], paddingData=[], shapeData=[], suffix='', blockLabels=None):
        
        #Store input variables internally
        self.dataType = dataType
        self.blockNames = blockNames
        self.blockFilenames = blockFilenames
        self.blockSampleNames = blockSampleNames
        self.blockLocations = blockLocations
        self.sampleNames = sampleNames
        self.WSIFilenames = WSIFilenames
        self.WSILabels = WSILabels
        self.cropData = cropData
        self.paddingData = paddingData
        self.shapeData = shapeData
        self.suffix = suffix
        self.blockLabels = blockLabels
        
        
        
        #Specify internal object data/directories according to data type
        if dataType == 'blocks':
            self.dir_results = dir_blocks_results
            self.dir_visuals_labelGrids = dir_blocks_visuals_labelGrids
            self.dir_visuals_overlaidLabelGrids = dir_blocks_visuals_overlaidLabelGrids
            self.dir_results_labelGrids = dir_blocks_results_labelGrids
            self.dir_results_overlaidLabelGrids = dir_blocks_results_overlaidLabelGrids
            self.visualizeLabelGrids = visualizeLabelGrids_blocks
            self.visualizePredictionGrids = visualizePredictionGrids_blocks
        elif 'WSI' in dataType or dataType == 'recon':
            self.dir_results = dir_WSI_results
            self.dir_results_labelGrids = dir_WSI_results_labelGrids
            self.dir_results_overlaidLabelGrids = dir_WSI_results_overlaidLabelGrids
            self.visualizeLabelGrids = False
            self.visualizePredictionGrids = visualizePredictionGrids_WSI
        else:
            sys.exit('\nError - Unknown data type used when creating classifier object.')
        
        #Prepare computation environment
        self.device = f"cuda:{gpus[-1]}" if len(gpus) > 0 else "cpu"
        self.torchDevice = torch.device(self.device)
        
        #Prepare data for input through PyTorch model
        #Using num_workers>1 in the DataLoader objects causes bizzare semaphore/lock/descriptor issues/warnings; still appears to work, but keeping to 1 for safety
        self.blockData = DataPreprocessing_Classifier(self.blockFilenames, resizeSize_blocks)
        self.blockDataloader = DataLoader(self.blockData, batch_size=batchsizeClassifier, num_workers=1, shuffle=False, pin_memory=True)
        self.numBlockData = len(self.blockDataloader)
        self.numWSIData = len(self.WSIDataloader)
        
    #Classify input data
    def predict(self, inputData):
        return self.model(inputData)
    
    #Train model
    def train(self):
        #self.model = 
        return None
        
    #Evaluate model
    def evaluate(self):
    
        #Save results to disk
        dataPrintout, dataPrintoutNames = [blockNames, blockLabels, blockPredictions], ['Names', 'Labels', 'Raw Predictions']
        dataPrintout = pd.DataFrame(np.asarray(dataPrintout)).transpose()
        dataPrintout.columns=dataPrintoutNames
        dataPrintout.to_csv(self.dir_results + 'predictions_blocks.csv', index=False)
        
        #Evaluate per-block results
        computeClassificationMetrics(blockLabels, blockPredictions, self.dir_results, '_blocks_')
    
        #Classify each WSI, that had its component blocks classified, according to specified threshold of allowable malignant blocks
        sampleLabels, samplePredictions = [], []
        for sampleIndex, sampleName in enumerate(sampleNames):
            blockIndices = np.where(blockSampleNames == sampleName)[0]
            if thresholdWSI_GT: sampleLabels.append((np.mean(blockLabels[blockIndices]) >= thresholdWSI)*1)
            samplePredictions.append((np.mean(blockPredictions[blockIndices]) >= thresholdWSI)*1)
        
        #Convert list of WSI labels/predictions to arrays
        if thresholdWSI_GT: 
            sampleLabels = np.asarray(sampleLabels)
            mismatchedSamples = sampleNames[np.where(sampleLabels-WSILabels != 0)[0]].tolist()
            if len(mismatchedSamples) > 0: print('\nWarning - Determination of ground-truth labels for WSI using threshold method did not match with WSI labels in recorded metadata for the following samples: ' + str(mismatchedSamples)        else: sampleLabels = WSILabels
        samplePredictions = np.asarray(samplePredictions)
        
        #Evaluate per-sample results
        self.processResultsWSI(sampleNames, WSIFilenames, sampleLabels, samplePredictions, blockSampleNames, labels, blockPredictions, blockLocations)
    
    #Export model components (onnx for C# and a pythonic option)
    def exportClassifier(self):
        
        #Convert model to onnx and save to: dir_classifier_models + 'model_XGBClassifier.onnx'
        #model_onnx = self.model
        
        #Clear the model from memory
        del model_onnx
        if len(gpus) > 0: torch.cuda.empty_cache() 
        
        #Store the torch model across multiple 100 Mb files to bypass Github upload file size limits
        modelName = 'name'
        modelPath = dir_classifier_models + modelName
        torch.save(self.model, modelPath + '.pt')
        if os.path.exists(modelPath): shutil.rmtree(modelPath)
        os.makedirs(modelPath)
        with multivolumefile.open(modelPath + os.path.sep + modelName + '.7z', mode='wb', volume=104857600) as modelArchive:
            with py7zr.SevenZipFile(modelArchive, 'w') as archive:
                archive.writeall(modelPath + '.pt', modelName + '.pt')
        os.remove(modelPath + '.pt')
        
        #Clear the model from memory
        del self.model
        if len(gpus) > 0: torch.cuda.empty_cache() 
        
    #Load a pretrained model; automatically handles 100 Mb file compression
    #Should not require modification
    def loadClassifier(self):
        modelPath = modelDirectory + modelName
        with multivolumefile.open(modelPath + os.path.sep + modelName + '.7z', mode='rb') as modelArchive:
            with py7zr.SevenZipFile(modelArchive, 'r') as archive:
                archive.extract(modelDirectory)
        _ = self.model.load_state_dict(torch.load(modelPath + '.pt', map_location='cpu'))
        _ = self.model.train(False)
        os.remove(modelPath + '.pt')
    
    #Store results and visualizations of such to disk
    #Should not require modification
    def processResultsWSI(self, sampleNames, WSIFilenames, sampleLabels, samplePredictions, blockSampleNames, blockLabels, blockPredictions, blockLocations):
        
        #Save results to disk
        dataPrintout, dataPrintoutNames = [sampleNames, sampleLabels, samplePredictions], ['Names', 'Labels', 'Raw Predictions']
        dataPrintout = pd.DataFrame(np.asarray(dataPrintout)).transpose()
        dataPrintout.columns=dataPrintoutNames
        dataPrintout.to_csv(self.dir_results + 'predictions_WSI.csv', index=False)
        
        #Evaluate WSI results
        computeClassificationMetrics(sampleLabels, samplePredictions, self.dir_results, '_WSI')
        
        #If the labels/predictions should be mapped visually onto the WSI
        if self.visualizeLabelGrids or self.visualizePredictionGrids:
            for sampleIndex, sampleName in tqdm(enumerate(sampleNames), total=len(sampleNames), desc='Result Visualization', leave=True, ascii=asciiFlag):
                
                #Get indices for the sample blocks
                blockIndices = np.where(blockSampleNames == sampleName)[0]
                
                #Load the sample WSI
                imageWSI = cv2.cvtColor(cv2.imread(WSIFilenames[sampleIndex], cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB)
                if (len(self.cropData) != 0): 
                    cropData, paddingData = self.cropData[sampleIndex], self.paddingData[sampleIndex]
                    imageWSI = np.pad(imageWSI[cropData[0]:cropData[1], cropData[2]:cropData[3]], ((paddingData[0], paddingData[1]), (paddingData[2], paddingData[3]), (0, 0)))
                
                #Create grid overlays
                if self.visualizePredictionGrids:
                    gridOverlay_Predictions = np.zeros(imageWSI.shape, dtype=np.uint8)
                    colorsPredictions = (cmapClasses(blockPredictions)[:,:3].astype(np.uint8)*255).tolist()
                
                if self.visualizeLabelGrids: 
                    gridOverlay_GT = np.zeros(imageWSI.shape, dtype=np.uint8)
                    colorsLabels = (cmapClasses(blockLabels)[:,:3].astype(np.uint8)*255).tolist()
                for blockIndex in blockIndices:
                    startRow, startColumn = blockLocations[blockIndex] 
                    posStart, posEnd = (startColumn, startRow), (startColumn+blockSize, startRow+blockSize)
                    if self.visualizeLabelGrids: gridOverlay_GT = rectangle(gridOverlay_GT, posStart, posEnd, colorsLabels[blockIndex])
                    if self.visualizePredictionGrids: gridOverlay_Predictions = rectangle(gridOverlay_Predictions, posStart, posEnd, colorsPredictions[blockIndex])
                    
                #Store overlays to disk
                if self.visualizeLabelGrids: exportImage(self.dir_visuals_labelGrids+'overlay_labelGrid_'+sampleName+overlayExtension, gridOverlay_GT)
                if self.visualizePredictionGrids:
                    exportImage(self.dir_results_labelGrids+'overlay_predictionsGrid_'+sampleName+overlayExtension, gridOverlay_Predictions)
                
                #Overlay grids on top of WSI and store to disk
                if self.visualizeLabelGrids: 
                    imageWSI_GT = cv2.addWeighted(imageWSI, 1.0, gridOverlay_GT, overlayWeight, 0.0)
                    exportImage(self.dir_visuals_overlaidLabelGrids+'labelGrid_'+sampleName+overlayExtension, imageWSI_GT)
                if self.visualizePredictionGrids: 
                    imageWSI_Predictions = cv2.addWeighted(imageWSI, 1.0, gridOverlay_Predictions, overlayWeight, 0.0)
                    exportImage(self.dir_results_overlaidLabelGrids+'predictionsGrid_'+sampleName+overlayExtension, imageWSI_Predictions)
    
    #Classify a sample WSI and generate data for reconstruction model
    #Should not require modification
    def classifyWSI(self, evaluatePredictions):
        
        dataInput = np.asarray(self.blockFeatures.astype(np.float32))
        blockPredictions = self.predict(dataInput)
        
        #Clear the model from memory
        del self.model
        if len(gpus) > 0: torch.cuda.empty_cache() 
        
        #Classify each WSI, that had its component blocks classified, according to specified threshold of allowable malignant blocks
        sampleLabels, samplePredictions, sampleBlockIndices = [], [], [], []
        for sampleIndex, sampleName in enumerate(self.sampleNames):
            blockIndices = np.where(self.blockSampleNames == sampleName)[0]
            sampleBlockIndices.append(blockIndices)
            samplePredictions.append((np.mean(blockPredictions[blockIndices]) >= thresholdWSI)*1)
        samplePredictions = np.asarray(samplePredictions)
        
        #Evaluate per-sample results
        self.processResultsWSI(self.sampleNames, self.WSIFilenames, self.WSILabels, samplePredictions, self.blockSampleNames, [], blockPredictions, self.blockLocations)
        
        #Create prediction arrays for reconstruction model and save them to disk
        if classifierRecon:
            predictionMaps = []
            for sampleIndex, sampleName in tqdm(enumerate(self.sampleNames), total=len(self.sampleNames), desc='Data Assembly', leave=True, ascii=asciiFlag):
                blockIndices = sampleBlockIndices[sampleIndex]
                predictionLocations = self.blockLocations[blockIndices]//blockSize
                predictionMap = np.full((self.shapeData[sampleIndex]), valueBackground)
                predictionMap[predictionLocations[:,0], predictionLocations[:,1]] = blockPredictions[blockIndices]
                predictionMaps.append(predictionMap)
                if visualizeInputData_recon: exportImage(dir_recon_visuals_inputData+'predictionMap_'+sampleName+'.tif', cmapClasses(predictionMap)[:,:,:3].astype(np.uint8)*255)
            predictionMaps = np.asarray(predictionMaps, dtype='object')
            np.save(dir_recon_inputData + 'predictionMaps', predictionMaps)
        