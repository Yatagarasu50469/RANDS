#==================================================================
#MODEL_CLASS_ORIGINAL
#==================================================================

#Load and preprocess data image files
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
    def __init__(self, dataType, patchNames, patchFilenames, patchSampleNames, patchLocations, sampleNames, WSIFilenames, cropData=[], paddingData=[], shapeData=[], suffix='', WSILabels=None, patchLabels=None):
        
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
        self.WSILabels = WSILabels
        self.patchLabels = patchLabels
        
        #Prepare data objects for obtaining/processing PyTorch model inputs
        self.device = f"cuda:{gpus[-1]}" if len(gpus) > 0 else "cpu"
        self.torchDevice = torch.device(self.device)
        self.patchData = DataPreprocessing_Classifier(self.patchFilenames, resizeSize_patches)
        self.patchDataloader = DataLoader(self.patchData, batch_size=batchsizeClassifier, num_workers=numberCPUS, shuffle=False, pin_memory=True)
        self.numPatchData = len(self.patchDataloader)
        self.WSIData = DataPreprocessing_Classifier(self.WSIFilenames, resizeSize_WSI)
        self.WSIDataloader = DataLoader(self.WSIData, batch_size=batchsizeClassifier, num_workers=numberCPUS, shuffle=False, pin_memory=True)
        self.numWSIData = len(self.WSIDataloader)
        
        #Set default cuda device for XGBClassifier input data
        if len(gpus) > 0: cp.cuda.Device(gpus[-1]).use()
        
        #Specify internal object data/directories according to data type
        if dataType == 'patches':
            self.overwrite_features = overwrite_patches_features
            self.overwrite_saliencyMaps = overwrite_patches_saliencyMaps
            self.visualizeSaliencyMaps = visualizeSaliencyMaps_patches
            self.visualizeLabelGrids = visualizeLabelGrids_patches
            self.visualizePredictionGrids = visualizePredictionGrids_patches
            self.dir_results = dir_patches_results
            self.dir_features = dir_patches_features
            self.dir_saliencyMaps = dir_patches_salicencyMaps
            self.dir_visuals_saliencyMaps = dir_patches_visuals_saliencyMaps
            self.dir_visuals_overlaidSaliencyMaps = dir_patches_visuals_overlaidSaliencyMaps
            self.dir_visuals_labelGrids = dir_patches_visuals_labelGrids
            self.dir_visuals_overlaidLabelGrids = dir_patches_visuals_overlaidLabelGrids
            self.dir_visuals_predictionGrids = dir_patches_visuals_predictionGrids
            self.dir_visuals_overlaidPredictionGrids = dir_patches_visuals_overlaidPredictionGrids
            self.dir_visuals_fusionGrids = dir_patches_visuals_fusionGrids
            self.dir_visuals_overlaidFusionGrids = dir_patches_visuals_overlaidFusionGrids
        elif dataType == 'recon':
            self.overwrite_features = overwrite_recon_features
            self.overwrite_saliencyMaps = overwrite_recon_saliencyMaps
            self.visualizeSaliencyMaps = visualizeSaliencyMaps_recon
            self.visualizeLabelGrids = False
            self.visualizePredictionGrids = visualizePredictionGrids_recon
            self.dir_results = dir_recon_inputData
            self.dir_features = dir_recon_features
            self.dir_saliencyMaps = dir_recon_saliencyMaps
            self.dir_visuals_saliencyMaps = dir_recon_visuals_saliencyMaps
            self.dir_visuals_overlaidSaliencyMaps = dir_recon_visuals_overlaidSaliencyMaps
            self.dir_visuals_labelGrids = None
            self.dir_visuals_overlaidLabelGrids = None
            self.dir_visuals_predictionGrids = dir_recon_visuals_predictionGrids
            self.dir_visuals_overlaidPredictionGrids = dir_recon_visuals_overlaidPredictionGrids
            self.dir_visuals_fusionGrids = dir_recon_visuals_fusionGrids
            self.dir_visuals_overlaidFusionGrids = dir_recon_visuals_overlaidFusionGrids
        else:
            sys.exit('\nError - Unknown data type used when creating classifier object.\n')
        
        #Disable label grid visualization if there are no patch-level labels
        if len(patchLabels) == 0: self.visualizeLabelGrids = False
        
        #Extract or load features for the indicated patch files
        if self.overwrite_features: self.computeFeatures()
        else: self.patchFeatures = np.load(self.dir_features + 'patchFeatures'+self.suffix+'.npy', allow_pickle=True)
        
        #Extract or load saliency map data for the indicated patch files
        if evaluateMethodWSI == 'gradcam++':
            if self.overwrite_saliencyMaps: self.computeSalicencyMaps()
            else: self.patchWeights = np.load(self.dir_saliencyMaps + 'patchWeights'+self.suffix+'.npy', allow_pickle=True)
            
    def computeFeatures(self):
        
        #Reset RNG before setting up model; else initialization values may be be inconsistent
        resetRandom()
        
        #Load pretrained ResNet50 model and set to evaluation mode
        model_ResNet = models.resnet50(weights=weightsResNet).to(self.torchDevice)
        _ = model_ResNet.train(False)

        #Extract features for each batch of sample patch images
        self.patchFeatures = []
        for data in tqdm(self.patchDataloader, total=self.numPatchData, desc='Feature Determination', leave=True, ascii=asciiFlag):
            self.patchFeatures += model_ResNet(data.to(self.torchDevice)).detach().cpu().tolist()
        
        #Clear the ResNet model
        del model_ResNet
        if len(gpus) > 0: torch.cuda.empty_cache()
        
        #Convert list of features to an array
        self.patchFeatures = np.asarray(self.patchFeatures)
        
        #Save features to disk
        np.save(self.dir_features + 'patchFeatures'+self.suffix, self.patchFeatures)
    
    def computeSalicencyMaps(self):
        
        #Reset RNG before setting up model; else initialization values may be be inconsistent
        resetRandom()
        
        #Load pre-trained DenseNet
        model_DenseNet = models.densenet169(weights=weightsDenseNet)
        
        #Replace the in-built classifier; unclear how this structure and these hyperparameters were determined
        model_DenseNet.classifier = nn.Sequential(
            nn.Linear(model_DenseNet.classifier.in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 2),
            nn.LogSoftmax(dim=1)
        )
        model_DenseNet = model_DenseNet.to(self.torchDevice)
        _ = model_DenseNet.train(False)
        
        #Create GradCAMPlusPlus model; see https://github.com/jacobgil/pytorch-grad-cam for additional models and options
        model_GradCamPlusPlus = GradCAMPlusPlus(model=model_DenseNet, target_layers=[model_DenseNet.features[-1]])
        
        #Extract features for each batch of sample patch images
        #For current GradCamPlusPlus implementation, must manually clear internal copy of the outputs from GPU cache to prevent OOM
        #Do not need to move data to device here, as managed by GradCAMPlusPlus (having already been placed on device)
        self.saliencyMaps = []
        for data in tqdm(self.WSIDataloader, total=self.numWSIData, desc='Saliency Mapping', leave=True, ascii=asciiFlag):
            self.saliencyMaps.append(model_GradCamPlusPlus(input_tensor=data, targets=None))
            del model_GradCamPlusPlus.outputs
            if len(gpus) > 0: torch.cuda.empty_cache()
        self.saliencyMaps = np.vstack(self.saliencyMaps)
        
        #Clear DenseNet and GradCamPlusPlus
        del model_DenseNet, model_GradCamPlusPlus
        if len(gpus) > 0: torch.cuda.empty_cache()
        
        #Process each of the resulting maps
        self.patchWeights = []
        for index, saliencyMap in tqdm(enumerate(self.saliencyMaps), total=len(self.saliencyMaps), desc='Saliency Processing', leave=True, ascii=asciiFlag): 
            
            #If visualizing, store the saliency map to disk and load WSI; regardless, obtain WSI size
            if self.visualizeSaliencyMaps:
                _ = exportImage(self.dir_visuals_saliencyMaps+'saliencyMap_'+self.sampleNames[index], matplotlib.cm.jet(saliencyMap)[:,:,:-1].astype(np.float32), True)
                if overlayGray: overlaid = np.expand_dims(cv2.cvtColor(cv2.imread(self.WSIFilenames[index], cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2GRAY), -1)
                else: overlaid = cv2.cvtColor(cv2.imread(self.WSIFilenames[index], cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB)
                resizeShape = overlaid.shape[:2]
            else: 
                resizeShape = cv2.imread(self.WSIFilenames[index], 0).shape[:2]
            
            #Resize the saliency map to match the WSI dimensions
            transform = generateTransform(resizeShape, False, False)
            saliencyMap = transform(np.expand_dims(saliencyMap, 0))[0].numpy()
            
            #If visualizing, overlay the saliency map on the WSI and save it to disk
            if self.visualizeSaliencyMaps:
                transform = generateTransform([], True, False)
                overlaid = np.moveaxis(transform(np.moveaxis(overlaid, -1, 0)).numpy(), 0, -1)
                overlaid = show_cam_on_image(overlaid, saliencyMap, use_rgb=True, colormap=cv2.COLORMAP_HOT, image_weight=1.0-overlayWeight)
                _ = exportImage(self.dir_visuals_overlaidSaliencyMaps+'overlaidSaliency_'+self.sampleNames[index], overlaid, exportLossless)
            
            #Extract saliency map data specific to sample patch locations, compute regional importance as the average value, and threshold to get weights
            patchWeights = []
            for locationIndex, locationData in enumerate(self.patchLocations[np.where(self.patchSampleNames == self.sampleNames[index])[0]]):
                startRow, startColumn = locationData
                patchWeight = np.mean(saliencyMap[startRow:startRow+patchSize, startColumn:startColumn+patchSize])
                if patchWeight < 0.25: patchWeights.append(0)
                else: patchWeights.append(patchWeight)
            self.patchWeights += patchWeights
        
        #Convert list of patch weights to an array and save to disk
        self.patchWeights = np.asarray(self.patchWeights)
        np.save(self.dir_saliencyMaps + 'patchWeights'+self.suffix, self.patchWeights)
    
    #Classify extracted patch features, obtaining raw patch predictions and then performing decision fusion
    def predict(self, inputs, weights=None):
        predictions = self.model_XGBClassifier.predict(inputs.astype(np.float32))
        if evaluateMethodWSI == 'gradcam++': 
            predictionsFusion = np.where(predictions==0, -1, 1)*weights
            predictionsFusion = np.where(predictionsFusion>0, 1, 0)
            return predictions.tolist(), predictionsFusion.tolist()
        elif evaluateMethodWSI == 'majority':
            return predictions.tolist(), []
        else: 
            sys.exit('Error - Unknown evaluateMethodWSI method provided: ' + evaluateMethodWSI)
    
    #Classify WSI, that had component patches classified, according ratio of malignant to foreground patches
    def classifyWSI(self, sampleNames, WSIFilenames, patchSampleNames, patchPredictions, patchPredictionsFusion, patchLocations): 
        samplePredictions, samplePredictionsFusion, samplePatchIndices = [], [], []
        for sampleIndex, sampleName in enumerate(sampleNames):
            patchIndices = np.where(patchSampleNames == sampleName)[0]
            samplePatchIndices.append(patchIndices)
            if thresholdWSI_prediction == 0:
                if np.sum(patchPredictions[patchIndices]) > 0: patchPredictions.append(1)
                else: patchPredictions.append(0)
                if evaluateMethodWSI == 'gradcam++': 
                    if np.sum(patchPredictionsFusion[patchIndices]) > 0: samplePredictionsFusion.append(1)
                    else: samplePredictionsFusion.append(0)
            else: 
                if np.mean(patchPredictions[patchIndices]) >= thresholdWSI_prediction: samplePredictions.append(1)
                else: samplePredictions.append(0)
                if evaluateMethodWSI == 'gradcam++': 
                    if np.mean(patchPredictionsFusion[patchIndices]) >= thresholdWSI_prediction: samplePredictionsFusion.append(1)
                    else: samplePredictionsFusion.append(0)
        return np.asarray(samplePredictions), np.asarray(samplePredictionsFusion), samplePatchIndices
    
    #Perform cross-validation
    def crossValidation(self):
        
        #Allocate samples to folds for cross validation
        if type(manualFolds) != list: folds = [array.tolist() for array in np.array_split(np.random.permutation(self.sampleNames), manualFolds)]
        else: folds = manualFolds
        numFolds = len(folds)

        #Split patch features into specified folds, keeping track of originating indices and matched labels
        foldsFeatures, foldsLabels, foldsWeights, foldsPatchSampleNames, foldsPatchNames, foldsPatchLocations = [], [], [], [], [], []
        for fold in folds:
            patchIndices = np.concatenate([np.where(self.patchSampleNames == sampleName)[0] for sampleName in fold])
            foldsPatchLocations.append(list(self.patchLocations[patchIndices]))
            foldsFeatures.append(list(self.patchFeatures[patchIndices]))
            foldsLabels.append(list(self.patchLabels[patchIndices]))
            if evaluateMethodWSI == 'gradcam++': foldsWeights.append(list(self.patchWeights[patchIndices]))
            else: foldsWeights.append([])
            foldsPatchSampleNames.append(list(self.patchSampleNames[patchIndices]))
            foldsPatchNames.append(list(self.patchNames[patchIndices]))            
        
        #Collapse data for later (correct/matched ordered) evaluation of the fold data
        foldsSampleNames = np.asarray(sum(folds, []))
        foldsPatchLocations = np.concatenate(foldsPatchLocations)
        foldsPatchSampleNames = np.concatenate(foldsPatchSampleNames)
        foldsPatchNames = np.concatenate(foldsPatchNames)
        foldsWSIFilenames = np.concatenate([self.WSIFilenames[np.where(self.sampleNames == sampleName)[0]] for sampleName in foldsSampleNames])
        foldsSampleLabels = np.concatenate([self.WSILabels[np.where(self.sampleNames == sampleName)[0]] for sampleName in foldsSampleNames])
        
        #Check class distribution between the folds
        #print('B\t M \t Total')
        #for foldNum in range(0, numFolds):
        #    print(np.sum(np.array(foldsLabels[foldNum]) == valueBenign),'\t', np.sum(np.array(foldsLabels[foldNum]) == valueMalignant), '\t', len(foldsLabels[foldNum]))
        
        #Perform training/testing among the folds, testing sequentially (to ensure the correct order) and storing results for later evaluation
        foldsPatchPredictions, foldsPatchPredictionsFusion = [], []
        for foldNum in tqdm(range(0, numFolds), desc='Patch Classification', leave=True, ascii=asciiFlag):
            
            #Train on all folds except the one specified
            self.train(np.concatenate(foldsFeatures[:foldNum]+foldsFeatures[foldNum+1:]), np.concatenate(foldsLabels[:foldNum]+foldsLabels[foldNum+1:]))
            
            #Extract the testing data and place on the GPU if able
            dataInput = np.asarray(foldsFeatures[foldNum])
            if len(gpus) > 0: dataInput = cp.asarray(dataInput)
            
            #Classify patches in the specified, remaining fold
            predictions, predictionsFusion = self.predict(dataInput, np.asarray(foldsWeights[foldNum]))
            foldsPatchPredictionsFusion += predictionsFusion
            foldsPatchPredictions += predictions
            
            #Clear the XGBClassifier model and data on GPU
            del self.model_XGBClassifier, dataInput
            if len(gpus) > 0: 
                torch.cuda.empty_cache() 
                cp._default_memory_pool.free_all_blocks()
        
        #Convert lists of predictions to arrays and collapse labels for evaluation
        foldsLabels = np.concatenate(foldsLabels)
        foldsPatchPredictions = np.asarray(foldsPatchPredictions)
        foldsPatchPredictionsFusion = np.asarray(foldsPatchPredictionsFusion)
        
        #Classify WSI
        foldsSamplePredictions, foldsSamplePredictionsFusion, _ = self.classifyWSI(foldsSampleNames, foldsWSIFilenames, foldsPatchSampleNames, foldsPatchPredictions, foldsPatchPredictionsFusion, foldsPatchLocations)
        
        #Evaluate per-sample results
        self.processResultsWSI(foldsSampleNames, foldsWSIFilenames, foldsSampleLabels, foldsSamplePredictions, foldsSamplePredictionsFusion, foldsPatchSampleNames, foldsPatchNames, foldsLabels, foldsPatchPredictions, foldsPatchPredictionsFusion, foldsPatchLocations)
    
    def processResultsWSI(self, sampleNames, WSIFilenames, sampleLabels, samplePredictions, samplePredictionsFusion, patchSampleNames, patchNames, patchLabels, patchPredictions, patchPredictionsFusion, patchLocations):
        
        #Save patch results to disk
        dataPrintout, dataPrintoutNames = [patchNames], ['Names']
        if len(patchLabels) > 0: 
            dataPrintout.append(patchLabels)
            dataPrintoutNames.append('Labels')
        dataPrintout.append(patchPredictions)
        dataPrintoutNames.append('Initial Predictions')
        if evaluateMethodWSI == 'gradcam++': 
            dataPrintout.append(patchPredictionsFusion)
            dataPrintoutNames.append('Fusion Predictions')
        dataPrintout = pd.DataFrame(np.asarray(dataPrintout)).transpose()
        dataPrintout.columns=dataPrintoutNames
        dataPrintout.to_csv(self.dir_results + 'predictions_patches.csv', index=False)
        
        #Save WSI results to disk
        dataPrintout, dataPrintoutNames = [sampleNames], ['Names']
        if len(sampleLabels) > 0: 
            dataPrintout.append(sampleLabels)
            dataPrintoutNames.append('Labels')
        dataPrintout.append(samplePredictions)
        dataPrintoutNames.append('majority Predictions')
        if evaluateMethodWSI == 'gradcam++':  
            dataPrintout.append(samplePredictionsFusion)
            dataPrintoutNames.append('Fusion Predictions')
        dataPrintout = pd.DataFrame(np.asarray(dataPrintout)).transpose()
        dataPrintout.columns=dataPrintoutNames
        dataPrintout.to_csv(self.dir_results + 'predictions_WSI.csv', index=False)
        
        #Evaluate patch/WSI results if labels are available
        if len(patchLabels) > 0: 
            if len(patchLabels) != len(patchPredictions): sys.exit('\nError - The number of patch labels does not match the number of predictions.\n')
            if evaluateMethodWSI == 'gradcam++' and len(patchLabels) != len(patchPredictionsFusion): sys.exit('\nError - The number of patch labels does not match the number of fusion predictions.\n')
            computeClassificationMetrics(patchLabels, patchPredictions, self.dir_results, '_patches_initial')
            if evaluateMethodWSI == 'gradcam++': computeClassificationMetrics(patchLabels, patchPredictionsFusion, self.dir_results, '_patches_fusion')
        if len(sampleLabels) > 0: 
            if evaluateMethodWSI == 'gradcam++' and len(sampleLabels) != len(samplePredictionsFusion): sys.exit('\nError - The number of WSI labels does not match the number of fusion predictions.\n')
            computeClassificationMetrics(sampleLabels, samplePredictions, self.dir_results, '_WSI_majority')
            if evaluateMethodWSI == 'gradcam++': computeClassificationMetrics(sampleLabels, samplePredictionsFusion, self.dir_results, '_WSI_fusion')
        
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
                    if evaluateMethodWSI == 'gradcam++': 
                        gridOverlay_PredictionsFusion = np.zeros(imageWSI.shape, dtype=np.uint8)
                        colorsFusion = (cmapClasses(patchPredictionsFusion)[:,:3].astype(np.uint8)*255).tolist()
                if self.visualizeLabelGrids: 
                    gridOverlay_GT = np.zeros(imageWSI.shape, dtype=np.uint8)
                    colorsLabels = (cmapClasses(patchLabels)[:,:3].astype(np.uint8)*255).tolist()
                for patchIndex in patchIndices:
                    startRow, startColumn = patchLocations[patchIndex] 
                    posStart, posEnd = (startColumn, startRow), (startColumn+patchSize, startRow+patchSize)
                    if self.visualizeLabelGrids: gridOverlay_GT = rectangle(gridOverlay_GT, posStart, posEnd, colorsLabels[patchIndex])
                    if self.visualizePredictionGrids:
                        gridOverlay_Predictions = rectangle(gridOverlay_Predictions, posStart, posEnd, colorsPredictions[patchIndex])
                        if evaluateMethodWSI == 'gradcam++':  gridOverlay_PredictionsFusion = rectangle(gridOverlay_PredictionsFusion, posStart, posEnd, colorsFusion[patchIndex])
                    
                #Overlay grids on top of WSI and store to disk
                if self.visualizeLabelGrids: 
                    imageWSI_GT = cv2.addWeighted(imageWSI, 1.0, gridOverlay_GT, overlayWeight, 0.0)
                    _ = exportImage(self.dir_visuals_overlaidLabelGrids+'overlaid_labelGrid_'+sampleName, imageWSI_GT, exportLossless)
                if self.visualizePredictionGrids: 
                    imageWSI_Predictions = cv2.addWeighted(imageWSI, 1.0, gridOverlay_Predictions, overlayWeight, 0.0)
                    _ = exportImage(self.dir_visuals_overlaidPredictionGrids+'overlaid_predictionsGrid_'+sampleName, imageWSI_Predictions, exportLossless)
                    if evaluateMethodWSI == 'gradcam++': 
                        imageWSI_PredictionsFusion = cv2.addWeighted(imageWSI, 1.0, gridOverlay_PredictionsFusion, overlayWeight, 0.0)
                        _ = exportImage(self.dir_visuals_overlaidFusionGrids+'overlaid_fusionGrid_'+sampleName, imageWSI_PredictionsFusion, exportLossless)
                
                #Store overlays to disk
                if self.visualizeLabelGrids: 
                    gridOverlay_GT = cv2.cvtColor(np.dstack((gridOverlay_GT, np.uint8((np.sum(gridOverlay_GT, axis=-1) > 0)*255))), cv2.COLOR_RGBA2BGRA)
                    _ = exportImage(self.dir_visuals_labelGrids+'labelGrid_'+sampleName, gridOverlay_GT, exportLossless)
                if self.visualizePredictionGrids:
                    gridOverlay_Predictions = cv2.cvtColor(np.dstack((gridOverlay_Predictions, np.uint8((np.sum(gridOverlay_Predictions, axis=-1) > 0)*255))), cv2.COLOR_RGBA2BGRA)
                    _ = exportImage(self.dir_visuals_predictionGrids+'predictionsGrid_'+sampleName, gridOverlay_Predictions, exportLossless)
                    if evaluateMethodWSI == 'gradcam++': 
                        gridOverlay_PredictionsFusion = cv2.cvtColor(np.dstack((gridOverlay_PredictionsFusion, np.uint8((np.sum(gridOverlay_PredictionsFusion, axis=-1) > 0)*255))), cv2.COLOR_RGBA2BGRA)
                        _ = exportImage(self.dir_visuals_fusionGrids+'fusionGrid_'+sampleName, gridOverlay_PredictionsFusion, exportLossless)
    
    #Train a new XGB Classifier model
    def train(self, inputs, labels):
        self.model_XGBClassifier = XGBClassifier(device=self.device, seed=0)
        _  = self.model_XGBClassifier.fit(inputs.astype(np.float32), labels)
    
    #Train on all available data and export models
    def exportClassifier(self):
        
        #Setup, train, save, and clear the XGBClassifier model
        self.train(self.patchFeatures, self.patchLabels)
        
        #Save to disk in .json format for easy reloading
        self.model_XGBClassifier.save_model(dir_classifier_models + 'model_XGBClassifier.json')
        
        #Clear memory
        del self.model_XGBClassifier#, model_onnx_XGBClassifier
        if len(gpus) > 0: 
            torch.cuda.empty_cache() 
            cp._default_memory_pool.free_all_blocks()
        
    #Load a pretrained model
    def loadClassifier(self):
        self.model_XGBClassifier = XGBClassifier()
        self.model_XGBClassifier.load_model(dir_classifier_models + 'model_XGBClassifier.json')
        self.model_XGBClassifier._Booster.set_param({'device': self.device})
        
    #Generate data for reconstruction model
    def generateReconData(self):
        
        #Place data on the GPU if able
        dataInput = np.asarray(self.patchFeatures.astype(np.float32))
        if len(gpus) > 0: dataInput = cp.asarray(dataInput)
        
        #Classify patches
        patchPredictions, patchPredictionsFusion = self.predict(dataInput, self.patchWeights)
        patchPredictions, patchPredictionsFusion = np.asarray(patchPredictions), np.asarray(patchPredictionsFusion)
        
        #Clear the XGBClassifier model and data on GPU
        del self.model_XGBClassifier, dataInput
        if len(gpus) > 0: 
            torch.cuda.empty_cache() 
            cp._default_memory_pool.free_all_blocks()
        
        #Classify WSI
        samplePredictions, samplePredictionsFusion, samplePatchIndices = self.classifyWSI(self.sampleNames, self.WSIFilenames, self.patchSampleNames, patchPredictions, patchPredictionsFusion, self.patchLocations)

        #Evaluate per-sample results
        self.processResultsWSI(self.sampleNames, self.WSIFilenames, [], samplePredictions, samplePredictionsFusion, self.patchSampleNames, self.patchNames, [], patchPredictions, patchPredictionsFusion, self.patchLocations)
        
        #Create prediction arrays for reconstruction model and save them to disk
        if classifierRecon:
            predictionMaps, predictionsFusionMaps = [], []
            for sampleIndex, sampleName in tqdm(enumerate(self.sampleNames), total=len(self.sampleNames), desc='Data Assembly', leave=True, ascii=asciiFlag):
                patchIndices = samplePatchIndices[sampleIndex]
                predictionLocations = self.patchLocations[patchIndices]//patchSize
                predictionMap = np.full((self.shapeData[sampleIndex]), valueBackground)
                predictionMap[predictionLocations[:,0], predictionLocations[:,1]] = patchPredictions[patchIndices]
                predictionMaps.append(predictionMap)
                if visualizeInputData_recon: _ = exportImage(dir_recon_visuals_inputData+'predictionMap_'+sampleName, cmapClasses(predictionMap)[:,:,:3].astype(np.uint8)*255, exportLossless)
                if evaluateMethodWSI == 'gradcam++': 
                    predictionFusionMap = np.full((self.shapeData[sampleIndex]), valueBackground)
                    predictionFusionMap[predictionLocations[:,0], predictionLocations[:,1]] = patchPredictionsFusion[patchIndices]
                    predictionsFusionMaps.append(predictionFusionMap)
                    if visualizeInputData_recon: _ = exportImage(dir_recon_visuals_inputData+'fusionMap_'+sampleName, cmapClasses(predictionFusionMap)[:,:,:3].astype(np.uint8)*255, exportLossless)
            predictionMaps = np.asarray(predictionMaps, dtype='object')
            np.save(dir_recon_inputData + 'predictionMaps', predictionMaps)
            if evaluateMethodWSI == 'gradcam++': 
                predictionsFusionMaps = np.asarray(predictionsFusionMaps, dtype='object')
                np.save(dir_recon_inputData + 'fusionMaps', predictionsFusionMaps)
                