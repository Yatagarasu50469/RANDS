#==================================================================
#MODEL_CLASS_UPDATED
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
    
    #Load patches specified by provided filenames, extract relevant features, and setup additional model components needed for training/evaluation
    def __init__(self, dataType, patchNames, patchFilenames, patchSampleNames, patchLocations, sampleNames, WSIFilenames, cropData=[], paddingData=[], shapeData=[], suffix='', patchLabels=None):
        
        #Store input variables internally
        self.dataType = dataType
        self.patchNames = patchNames
        self.patchFilenames = patchFilenames
        self.patchSampleNames = patchSampleNames
        self.patchLocations = patchLocations
        self.sampleNames = sampleNames
        self.WSIFilenames = WSIFilenames
        self.cropData = cropData
        self.paddingData = paddingData
        self.shapeData = shapeData
        self.suffix = suffix
        self.patchLabels = patchLabels
        
        #Prepare data objects for obtaining/processing PyTorch model inputs
        self.device = f"cuda:{gpus[-1]}" if len(gpus) > 0 else "cpu"
        self.torchDevice = torch.device(self.device)
        self.patchData = DataPreprocessing_Classifier(self.patchFilenames, resizeSize_patches)
        self.patchDataloader = DataLoader(self.patchData, batch_size=batchsizeClassifier, num_workers=numberCPUS, shuffle=False, pin_memory=True)
        self.numPatchData = len(self.patchDataloader)
        #self.WSIData = DataPreprocessing_Classifier(self.WSIFilenames, resizeSize_WSI)
        #self.WSIDataloader = DataLoader(self.WSIData, batch_size=batchsizeClassifier, num_workers=numberCPUS, shuffle=False, pin_memory=True)
        #self.numWSIData = len(self.WSIDataloader)
        
        #Specify internal object data/directories according to data type
        if dataType == 'patches':
            self.visualizeLabelGrids = visualizeLabelGrids_patches
            self.visualizePredictionGrids = visualizePredictionGrids_patches
            self.dir_results = dir_patches_results
            self.dir_visuals_labelGrids = dir_patches_visuals_labelGrids
            self.dir_visuals_overlaidLabelGrids = dir_patches_visuals_overlaidLabelGrids
            self.dir_visuals_predictionGrids = dir_patches_visuals_predictionGrids
            self.dir_visuals_overlaidPredictionGrids = dir_patches_visuals_overlaidPredictionGrids
        elif dataType == 'recon':
            self.visualizeLabelGrids = False
            self.visualizePredictionGrids = visualizePredictionGrids_recon
            self.dir_results = dir_recon_inputData
            self.dir_visuals_labelGrids = None
            self.dir_visuals_overlaidLabelGrids = None
            self.dir_visuals_predictionGrids = dir_recon_visuals_predictionGrids
            self.dir_visuals_overlaidPredictionGrids = dir_recon_visuals_overlaidPredictionGrids
        else:
            sys.exit('\nError - Unknown data type used when creating classifier object.\n')
    
    #Classify extracted patches
    def predict(self, inputs):
        return self.model(inputs)
    
    #Create and train a new classifier model
    def train(self, inputs, labels):
        #self.model = 
        return None
    
    #Evaluate model
    def evaluate(self):
    
        #Save results to disk
        dataPrintout, dataPrintoutNames = [self.patchNames, self.patchLabels, patchPredictions], ['Names', 'Labels', 'Predictions']
        dataPrintout = pd.DataFrame(np.asarray(dataPrintout)).transpose()
        dataPrintout.columns=dataPrintoutNames
        dataPrintout.to_csv(self.dir_results + 'predictions_patches.csv', index=False)
        
        #Evaluate per-patch results
        computeClassificationMetrics(self.patchLabels, patchPredictions, self.dir_results, '_patches_')
    
        #Classify WSI
        sampleLabels, samplePredictions, _ = self.classifyWSI(sampleNames, self.patchSampleNames, self.patchLabels, patchPredictions)
        
        #Evaluate per-sample results
        self.processResultsWSI(sampleNames, WSIFilenames, sampleLabels, samplePredictions, self.patchSampleNames, self.patchLabels, patchPredictions, self.patchLocations)
    
    #Classify WSI, that had component patches classified, according to specified threshold of allowable malignant patches
    def classifyWSI(self, sampleNames, patchSampleNames, patchLabels, patchPredictions): 
        sampleLabels, samplePredictions, samplePatchIndices = [], [], []
        for sampleIndex, sampleName in enumerate(sampleNames):
            patchIndices = np.where(patchSampleNames == sampleName)[0]
            samplePatchIndices.append(patchIndices)
            if len(patchLabels) > 0: sampleLabels.append((np.mean(patchLabels[patchIndices]) >= thresholdWSI)*1)
            samplePredictions.append((np.mean(patchPredictions[patchIndices]) >= thresholdWSI)*1)
        return np.asarray(sampleLabels), np.asarray(samplePredictions), samplePatchIndices
    
    #Process the WSI classification results
    def processResultsWSI(self, sampleNames, WSIFilenames, sampleLabels, samplePredictions, patchSampleNames, patchLabels, patchPredictions, patchLocations):
        
        #Save results to disk
        if len(sampleLabels) > 0: dataPrintout, dataPrintoutNames = [sampleNames, sampleLabels, samplePredictions], ['Names', 'Labels', 'Predictions']
        else: dataPrintout, dataPrintoutNames = [sampleNames, samplePredictions], ['Names', 'Predictions']
        dataPrintout = pd.DataFrame(np.asarray(dataPrintout)).transpose()
        dataPrintout.columns=dataPrintoutNames
        dataPrintout.to_csv(self.dir_results + 'predictions_WSI.csv', index=False)
        
        #Evaluate WSI results if labels/predictions are available
        if len(sampleLabels) > 0 and len(samplePredictions) > 0: 
            if len(sampleLabels) != len(samplePredictions): sys.exit('\nError - The number of WSI labels does not match the number of predictions.\n')
            computeClassificationMetrics(sampleLabels, samplePredictions, self.dir_results, '_WSI_initial')
        
        #If the labels/predictions should be mapped visually onto the WSI
        if self.visualizeLabelGrids or self.visualizePredictionGrids:
            for sampleIndex, sampleName in tqdm(enumerate(sampleNames), total=len(sampleNames), desc='Result Visualization', leave=True, ascii=asciiFlag):
                
                #Get indices for the sample patches
                patchIndices = np.where(patchSampleNames == sampleName)[0]
                
                #Load the sample WSI
                imageWSI = cv2.cvtColor(cv2.imread(WSIFilenames[sampleIndex], cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB)
                if (len(self.cropData) != 0): 
                    cropData, paddingData = self.cropData[sampleIndex], self.paddingData[sampleIndex]
                    imageWSI = np.pad(imageWSI[cropData[0]:cropData[1], cropData[2]:cropData[3]], ((paddingData[0], paddingData[1]), (paddingData[2], paddingData[3]), (0, 0)))
                
                #Create grid overlays
                if self.visualizePredictionGrids:
                    gridOverlay_Predictions = np.zeros(imageWSI.shape, dtype=np.uint8)
                    colorsPredictions = (cmapClasses(patchPredictions)[:,:3].astype(np.uint8)*255).tolist()
                if self.visualizeLabelGrids: 
                    gridOverlay_GT = np.zeros(imageWSI.shape, dtype=np.uint8)
                    colorsLabels = (cmapClasses(patchLabels)[:,:3].astype(np.uint8)*255).tolist()
                for patchIndex in patchIndices:
                    startRow, startColumn = patchLocations[patchIndex] 
                    posStart, posEnd = (startColumn, startRow), (startColumn+patchSize, startRow+patchSize)
                    if self.visualizeLabelGrids: gridOverlay_GT = rectangle(gridOverlay_GT, posStart, posEnd, colorsLabels[patchIndex])
                    if self.visualizePredictionGrids:
                        gridOverlay_Predictions = rectangle(gridOverlay_Predictions, posStart, posEnd, colorsPredictions[patchIndex])
                    
                #Store overlays to disk
                if self.visualizeLabelGrids: _ = exportImage(self.dir_visuals_labelGrids+'labelGrid_'+sampleName, gridOverlay_GT, exportLossless)
                if self.visualizePredictionGrids: _ = exportImage(self.dir_visuals_predictionGrids+'predictionsGrid_'+sampleName, gridOverlay_Predictions, exportLossless)
                
                #Overlay grids on top of WSI and store to disk
                if self.visualizeLabelGrids: 
                    imageWSI_GT = cv2.addWeighted(imageWSI, 1.0, gridOverlay_GT, overlayWeight, 0.0)
                    _ = exportImage(self.dir_visuals_overlaidLabelGrids+'overlaid_labelGrid_'+sampleName, imageWSI_GT, exportLossless)
                if self.visualizePredictionGrids: 
                    imageWSI_Predictions = cv2.addWeighted(imageWSI, 1.0, gridOverlay_Predictions, overlayWeight, 0.0)
                    _ = exportImage(self.dir_visuals_overlaidPredictionGrids+'overlaid_predictionsGrid_'+sampleName, imageWSI_Predictions, exportLossless)

    #Generate data for reconstruction model training/testing
    def generateReconData(self):
        
        #Place data on the GPU if able
        dataInput = np.asarray(self.patchFeatures.astype(np.float32))
        if len(gpus) > 0: dataInput = cp.asarray(dataInput)
        
        #Classify patches
        patchPredictions = np.asarray(self.predict(dataInput))
        
        #Clear the model and data on GPU
        del self.model_XGBClassifier, dataInput
        if len(gpus) > 0: 
            torch.cuda.empty_cache() 
            cp._default_memory_pool.free_all_blocks()
        
        #Classify the WSI
        _, samplePredictions, samplePatchIndices = self.classifyWSI(self.sampleNames, self.patchSampleNames, [], patchPredictions)
        
        #Evaluate per-sample results
        self.processResultsWSI(self.sampleNames, self.WSIFilenames, [], samplePredictions, self.patchSampleNames, self.patchNames, [], patchPredictions, self.patchLocations)
        
        #Create prediction arrays for reconstruction model and save them to disk
        if classifierRecon:
            predictionMaps = []
            for sampleIndex, sampleName in tqdm(enumerate(self.sampleNames), total=len(self.sampleNames), desc='Data Assembly', leave=True, ascii=asciiFlag):
                patchIndices = samplePatchIndices[sampleIndex]
                predictionLocations = self.patchLocations[patchIndices]//patchSize
                predictionMap = np.full((self.shapeData[sampleIndex]), valueBackground)
                predictionMap[predictionLocations[:,0], predictionLocations[:,1]] = patchPredictions[patchIndices]
                predictionMaps.append(predictionMap)
                if visualizeInputData_recon: _ = exportImage(dir_recon_visuals_inputData+'predictionMap_'+sampleName, cmapClasses(predictionMap)[:,:,:3].astype(np.uint8)*255, exportLossless)
            predictionMaps = np.asarray(predictionMaps, dtype='object')
            np.save(dir_recon_inputData + 'predictionMaps', predictionMaps)

    #Export model components; 
    def exportClassifier(self):
        
        #Store the torch model across multiple 100 Mb files to bypass Github upload file size limits
        modelPath = dir_classifier_models + 'model_updated'
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
    def loadClassifier(self):
        modelPath = modelDirectory + 'model_updated'
        with multivolumefile.open(modelPath + os.path.sep + modelName + '.7z', mode='rb') as modelArchive:
            with py7zr.SevenZipFile(modelArchive, 'r') as archive:
                archive.extract(modelDirectory)
        _ = self.model.load_state_dict(torch.load(modelPath + '.pt', map_location='cpu'))
        _ = self.model.train(False)
        os.remove(modelPath + '.pt')