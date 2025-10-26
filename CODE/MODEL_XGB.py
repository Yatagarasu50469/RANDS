#==================================================================
#MODEL_CLASS_ORIGINAL
#==================================================================

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
        
        #Prepare data objects for obtaining/processing PyTorch model inputs; changing batchsize changes results due to order of operations
        self.device = f"cuda:{gpus[-1]}" if len(gpus) > 0 else "cpu"
        self.torchDevice = torch.device(self.device)
        self.patchData = DataPreprocessing_Classifier(self.patchFilenames, resizeSize_patches)
        self.patchDataloader = DataLoader(self.patchData, batch_size=batchsizeClassifier, num_workers=numWorkers_patches, shuffle=False, pin_memory=True)
        self.numPatchData = len(self.patchDataloader)
        self.WSIData = DataPreprocessing_Classifier(self.WSIFilenames, resizeSize_WSI)
        self.WSIDataloader = DataLoader(self.WSIData, batch_size=batchsizeClassifier, num_workers=numWorkers_WSI, shuffle=False, pin_memory=True)
        self.numWSIData = len(self.WSIDataloader)
        
        #Set default cuda device for XGBClassifier input data
        if len(gpus) > 0: cp.cuda.Device(gpus[-1]).use()
        
        #Specify internal object data/directories according to data type
        if dataType == 'known':
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
        elif dataType == 'unknown':
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
            sys.exit('\nError - Unknown data type: ' + dataType + ' used when creating classifier object.\n')
        
        #Disable label grid visualization if there are no patch-level labels
        if len(patchLabels) == 0: self.visualizeLabelGrids = False
        
        #Extract or load features for the indicated patch files; includes synthetic data (use self.patchFilenames for length)
        if not self.overwrite_features: 
            self.patchFeatures = np.load(self.dir_features + 'patchFeatures'+self.suffix+'.npy', allow_pickle=True)
            if len(self.patchFeatures) != len(self.patchFilenames): 
                print('\nWarning - Number of patch features did not match number of patches; re-computing features\n')
                self.overwrite_features = True
        if self.overwrite_features: self.computeFeatures()
        
        #Extract or load saliency map data for the indicated patch files; does not include synthetic data (use self.patchNames for length)
        if evaluateMethodWSI == 'gradcam++':
            if not self.overwrite_saliencyMaps: 
                self.patchWeights = np.load(self.dir_saliencyMaps + 'patchWeights'+self.suffix+'.npy', allow_pickle=True)
                if len(self.patchWeights) != len(self.patchNames): 
                    print('\nWarning - Number of patch weights did not match number of samples; re-computing saliency maps\n')
                    self.overwrite_saliencyMaps = True
            if self.overwrite_saliencyMaps: self.computeSalicencyMaps()
    
    def computeFeatures(self):
        
        #Reset RNG before setting up model; else initialization values may be be inconsistent
        resetRandom()
        
        #Load pretrained ResNet50 model, move it to chosen compute device, and set to evaluation mode
        model_ResNet = models.resnet50(weights=weightsResNet)
        model_ResNet = model_ResNet.to(self.torchDevice)
        _ = model_ResNet.train(False)
        
        #Extract features for each batch of sample patch images
        self.patchFeatures = []
        with torch.no_grad():
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
        
        #Move model to chosen compute device and set to evaluation mode
        model_DenseNet = model_DenseNet.to(self.torchDevice)
        _ = model_DenseNet.train(False)
        
        #Create GradCAMPlusPlus model; see https://github.com/jacobgil/pytorch-grad-cam for additional models and options
        model_GradCamPlusPlus = GradCAMPlusPlus(model=model_DenseNet, target_layers=[model_DenseNet.features[-1]])
        
        #Extract features for each batch of sample patch images, must manually clear internal copy of the outputs from GPU cache to prevent OOM
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
            return predictions, predictionsFusion
        elif evaluateMethodWSI == 'majority':
            return predictions, np.asarray([])
        else: 
            sys.exit('Error - Unknown evaluateMethodWSI method provided: ' + evaluateMethodWSI)
    
    #Classify WSI, that had component patches classified, according ratio of malignant to foreground patches
    def classifyWSI(self, foldsSampleNames, foldsPatchSampleNames, foldsPatchPredictions, foldsPatchPredictionsFusion, numFolds=1): 
        foldsSamplePredictions, foldsSamplePredictionsFusion, foldsSamplePatchIndices = [], [], []
        for foldNum in range(0, numFolds): 
            if numFolds == 1: sampleNames, patchSampleNames, patchPredictions, patchPredictionsFusion = foldsSampleNames, foldsPatchSampleNames, foldsPatchPredictions, foldsPatchPredictionsFusion
            else: sampleNames, patchSampleNames, patchPredictions, patchPredictionsFusion = foldsSampleNames[foldNum], foldsPatchSampleNames[foldNum], foldsPatchPredictions[foldNum], foldsPatchPredictionsFusion[foldNum]
            samplePredictions, samplePredictionsFusion, samplePatchIndices = [], [], []
            for sampleIndex, sampleName in enumerate(sampleNames):
                patchIndices = np.where(patchSampleNames == sampleName)[0]
                samplePatchIndices.append(patchIndices)
                if thresholdWSI_prediction == 0:
                    if np.sum(patchPredictions[patchIndices]) > 0: samplePredictions.append(1)
                    else: samplePredictions.append(0)
                    if evaluateMethodWSI == 'gradcam++': 
                        if np.sum(patchPredictionsFusion[patchIndices]) > 0: samplePredictionsFusion.append(1)
                        else: samplePredictionsFusion.append(0)
                else: 
                    if np.mean(patchPredictions[patchIndices]) >= thresholdWSI_prediction: samplePredictions.append(1)
                    else: samplePredictions.append(0)
                    if evaluateMethodWSI == 'gradcam++': 
                        if np.mean(patchPredictionsFusion[patchIndices]) >= thresholdWSI_prediction: samplePredictionsFusion.append(1)
                        else: samplePredictionsFusion.append(0)
            foldsSamplePredictions.append(samplePredictions)
            foldsSamplePredictionsFusion.append(samplePredictionsFusion)
            foldsSamplePatchIndices.append(samplePatchIndices)
        
        return foldsSamplePredictions, foldsSamplePredictionsFusion, foldsSamplePatchIndices
    
    #Perform cross-validation
    def crossValidation(self):
        
        #Allocate samples to folds for cross validation
        if type(manualFolds) != list: foldsSampleNames = [array.tolist() for array in np.array_split(np.random.permutation(self.sampleNames), manualFolds)]
        else: foldsSampleNames = manualFolds
        numFolds = len(foldsSampleNames)
        
        #Split patch and WSI data according to specified folds
        foldsFeatures, foldsPatchLabels, foldsPatchWeights, foldsPatchSampleNames, foldsPatchNames, foldsPatchLocations, foldsWSIFilenames, foldsSampleLabels = [], [], [], [], [], [], [], []
        for fold in foldsSampleNames:
            indicesPatches = np.concatenate([np.where(self.patchSampleNames == sampleName)[0] for sampleName in fold])
            indicesWSI = np.concatenate([np.where(self.sampleNames == sampleName)[0] for sampleName in fold])
            foldsPatchLocations.append(list(self.patchLocations[indicesPatches]))
            foldsFeatures.append(list(self.patchFeatures[indicesPatches]))
            foldsPatchLabels.append(list(self.patchLabels[indicesPatches]))
            if evaluateMethodWSI == 'gradcam++': foldsPatchWeights.append(list(self.patchWeights[indicesPatches]))
            else: foldsPatchWeights.append([])
            foldsPatchSampleNames.append(self.patchSampleNames[indicesPatches])
            foldsPatchNames.append(list(self.patchNames[indicesPatches]))
            foldsWSIFilenames.append(list(self.WSIFilenames[indicesWSI]))
            foldsSampleLabels.append(list(self.WSILabels[indicesWSI]))
        
        #Extract synthetic patch data in list of lists to mimic fold, or empty list for an empty fold; always used in training
        indicesPatches = np.where(self.patchSampleNames == 'SYNTHETIC')[0]
        if len(indicesPatches) > 0: 
            features_synthetic, labelsPatches_synthetic = [list(self.patchFeatures[indicesPatches])], [list(self.patchLabels[indicesPatches])]
        else: features_synthetic, labelsPatches_synthetic = [], []
        
        #Perform training/testing among the folds, testing sequentially (to ensure the correct order) and storing results for later evaluation
        foldsPatchPredictions, foldsPatchPredictionsFusion = [], []
        for foldNum in tqdm(range(0, numFolds), desc='Patch Classification', leave=True, ascii=asciiFlag):
            
            #Train on all folds except the one specified
            self.train(np.concatenate(foldsFeatures[:foldNum]+foldsFeatures[foldNum+1:]+features_synthetic), np.concatenate(foldsPatchLabels[:foldNum]+foldsPatchLabels[foldNum+1:]+labelsPatches_synthetic))
            
            #Extract the testing data and place on the GPU if able
            dataInput = np.asarray(foldsFeatures[foldNum])
            if len(gpus) > 0: dataInput = cp.asarray(dataInput)
            
            #Classify patches in the specified, remaining fold
            predictions, predictionsFusion = self.predict(dataInput, np.asarray(foldsPatchWeights[foldNum]))
            foldsPatchPredictionsFusion.append(predictionsFusion)
            foldsPatchPredictions.append(predictions)
            
            #Clear the XGBClassifier model and data on GPU
            del self.model_XGBClassifier, dataInput
            if len(gpus) > 0: 
                torch.cuda.empty_cache() 
                cp._default_memory_pool.free_all_blocks()
        
        #Classify WSI
        foldsSamplePredictions, foldsSamplePredictionsFusion, _ = self.classifyWSI(foldsSampleNames, foldsPatchSampleNames, foldsPatchPredictions, foldsPatchPredictionsFusion, numFolds)
        
        #Evaluate per-sample results
        self.processResults(foldsSampleNames, foldsWSIFilenames, foldsSampleLabels, foldsSamplePredictions, foldsSamplePredictionsFusion, foldsPatchSampleNames, foldsPatchNames, foldsPatchLabels, foldsPatchPredictions, foldsPatchPredictionsFusion, foldsPatchLocations, numFolds)
    
    #Evaluate, export, and visualize classification results
    def processResults(self, foldsSampleNames, foldsWSIFilenames, foldsSampleLabels, foldsSamplePredictions, foldsSamplePredictionsFusion, foldsPatchSampleNames, foldsPatchNames, foldsPatchLabels, foldsPatchPredictions, foldsPatchPredictionsFusion, foldsPatchLocations, numFolds=1): 
        
        #Setup for printing data to disk
        dataPrintout_Patches, dataPrintout_WSI = {}, {}
        if numFolds > 1: dataPrintout_Patches['Fold'], dataPrintout_WSI['Fold'] = [], []
        dataPrintout_Patches['Names'], dataPrintout_WSI['Names'] = [], []
        if len(foldsPatchLabels) > 0: dataPrintout_Patches['Labels'] = []
        if len(foldsSampleLabels) > 0: dataPrintout_WSI['Labels'] = []
        dataPrintout_Patches['Initial Predictions'], dataPrintout_WSI['Majority Predictions'] = [], []
        if evaluateMethodWSI == 'gradcam++': dataPrintout_Patches['Fusion Predictions'],  dataPrintout_WSI['Fusion Predictions'] = [], []
        
        #Process results for each fold
        statisticsPatchesPredictions, statisticsPatchesPredictionsFusion, statisticsWSIPredictions, statisticsWSIPredictionsFusion = [], [], [], []
        for foldNum in range(0, numFolds): 
            
            #Isolate data for the fold and determine folder for output of fold-specific statistics if applicable
            if numFolds == 1: 
                sampleNames, WSIFilenames, sampleLabels, samplePredictions, samplePredictionsFusion, patchSampleNames, patchNames, patchLabels, patchPredictions, patchPredictionsFusion, patchLocations = foldsSampleNames, foldsWSIFilenames, foldsSampleLabels, foldsSamplePredictions, foldsSamplePredictionsFusion, foldsPatchSampleNames, foldsPatchNames, foldsPatchLabels, foldsPatchPredictions, foldsPatchPredictionsFusion, foldsPatchLocations
            else: 
                sampleNames, WSIFilenames, sampleLabels, samplePredictions, samplePredictionsFusion, patchSampleNames, patchNames, patchLabels, patchPredictions, patchPredictionsFusion, patchLocations = foldsSampleNames[foldNum], foldsWSIFilenames[foldNum], foldsSampleLabels[foldNum], foldsSamplePredictions[foldNum], foldsSamplePredictionsFusion[foldNum], foldsPatchSampleNames[foldNum], foldsPatchNames[foldNum], foldsPatchLabels[foldNum], foldsPatchPredictions[foldNum], foldsPatchPredictionsFusion[foldNum], foldsPatchLocations[foldNum]
            
            #Add patch-level information to printout
            if numFolds > 1: dataPrintout_Patches['Fold'] += (np.ones(len(patchNames), dtype=int)*(foldNum+1)).tolist()
            dataPrintout_Patches['Names'] += patchNames
            if len(patchLabels) > 0: dataPrintout_Patches['Labels'] += patchLabels
            dataPrintout_Patches['Initial Predictions'] += patchPredictions.tolist()
            if evaluateMethodWSI == 'gradcam++': dataPrintout_Patches['Fusion Predictions'] += patchPredictionsFusion.tolist()
            
            #Add WSI-level information to printout
            if numFolds > 1: dataPrintout_WSI['Fold'] += (np.ones(len(sampleNames), dtype=int)*(foldNum+1)).tolist()
            dataPrintout_WSI['Names'] += sampleNames
            if len(sampleLabels) > 0: dataPrintout_WSI['Labels'] += sampleLabels
            dataPrintout_WSI['Majority Predictions'] += samplePredictions
            if evaluateMethodWSI == 'gradcam++': dataPrintout_WSI['Fusion Predictions'] += samplePredictionsFusion
            
            #Evaluate patch/WSI results if labels are available
            if len(patchLabels) > 0: 
                statistics = computeClassificationMetrics(patchLabels, patchPredictions)                
                statisticsPatchesPredictions.append(statistics)
                if evaluateMethodWSI == 'gradcam++': 
                    statistics = computeClassificationMetrics(patchLabels, patchPredictionsFusion)
                    statisticsPatchesPredictionsFusion.append(statistics)
            if len(sampleLabels) > 0: 
                statistics = computeClassificationMetrics(sampleLabels, samplePredictions)
                statisticsWSIPredictions.append(statistics)
                if evaluateMethodWSI == 'gradcam++': 
                    statistics = computeClassificationMetrics(sampleLabels, samplePredictionsFusion)
                    statisticsWSIPredictionsFusion.append(statistics)
        
        #Save patch-level and WSI-level classification results to disk
        dataPrintout_Patches = pd.DataFrame.from_dict(dataPrintout_Patches)
        dataPrintout_Patches.to_csv(self.dir_results + 'predictions_patches.csv', index=False)
        dataPrintout_WSI = pd.DataFrame.from_dict(dataPrintout_WSI)
        dataPrintout_WSI.to_csv(self.dir_results + 'predictions_WSI.csv', index=False)
        
        #Save patch-level and WSI-level statistics to disk
        if len(patchLabels) > 0: 
            statisticsPatchesPredictions = consolidateStatistics(statisticsPatchesPredictions, numFolds)
            statisticsPatchesPredictions.to_csv(self.dir_results + 'statistics_patches_initial.csv', index=True, header=False)
            if evaluateMethodWSI == 'gradcam++': 
                statisticsPatchesPredictionsFusion = consolidateStatistics(statisticsPatchesPredictionsFusion, numFolds)
                statisticsPatchesPredictionsFusion.to_csv(self.dir_results + 'statistics_patches_fusion.csv', index=True, header=False)
        if len(sampleLabels) > 0: 
            statisticsWSIPredictions = consolidateStatistics(statisticsWSIPredictions, numFolds)
            statisticsWSIPredictions.to_csv(self.dir_results + 'statistics_WSI_majority.csv', index=True, header=False)
            if evaluateMethodWSI == 'gradcam++': 
                statisticsWSIPredictionsFusion = consolidateStatistics(statisticsWSIPredictionsFusion, numFolds)
                statisticsWSIPredictionsFusion.to_csv(self.dir_results + 'statistics_WSI_fusion.csv', index=True, header=False)
        
        #Flatten results for result visualization
        sampleNames, WSIFilenames, sampleLabels, samplePredictions, samplePredictionsFusion, patchSampleNames, patchLabels, patchPredictions, patchPredictionsFusion, patchLocations = np.concatenate(foldsSampleNames).tolist(), np.concatenate(foldsWSIFilenames).tolist(), np.concatenate(foldsSampleLabels).tolist(), np.concatenate(foldsSamplePredictions).tolist(), np.concatenate(foldsSamplePredictionsFusion).tolist(), np.concatenate(foldsPatchSampleNames), np.concatenate(foldsPatchLabels).tolist(), np.concatenate(foldsPatchPredictions).tolist(), np.concatenate(foldsPatchPredictionsFusion).tolist(), np.concatenate(foldsPatchLocations).tolist()

        #Visualize confusion matrices
        exportConfusionMatrix(patchLabels, patchPredictions, self.dir_results, '_patches_initial')
        exportConfusionMatrix(sampleLabels, samplePredictions, self.dir_results, '_WSI_majority')
        if evaluateMethodWSI == 'gradcam++': 
            exportConfusionMatrix(patchLabels, patchPredictionsFusion, self.dir_results, '_patches_fusion')
            exportConfusionMatrix(sampleLabels, samplePredictionsFusion, self.dir_results, '_WSI_fusion')
        
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
        self.model_XGBClassifier = XGBClassifier(device=self.device, random_state=manualSeedValue)
        _  = self.model_XGBClassifier.fit(inputs.astype(np.float32), labels)
    
    #Train on all available data and export models
    def exportClassifier(self):
        
        #Setup, train, save, and clear the XGBClassifier model
        self.train(self.patchFeatures, self.patchLabels)
        
        #Save to disk in .json format for easy reloading
        self.model_XGBClassifier.save_model(dir_classifier_models + 'model_XGBClassifier.json')
        
        #Clear memory
        del self.model_XGBClassifier
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
        samplePredictions, samplePredictionsFusion, samplePatchIndices = self.classifyWSI(self.sampleNames, self.WSIFilenames, self.patchSampleNames, patchPredictions, patchPredictionsFusion)

        #Evaluate per-sample results
        self.processResults(self.sampleNames, [], samplePredictions, samplePredictionsFusion, self.patchSampleNames, self.patchNames, [], patchPredictions, patchPredictionsFusion, self.patchLocations)
        
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

#Load and preprocess data; do not use/reference list objects, which cause pseudo memory-leak (Ref: https://github.com/pytorch/pytorch/issues/13246#issuecomment-905703662)
class DataPreprocessing_Classifier(Dataset):
    def __init__(self, filenames, resizeSize):
        super().__init__()
        self.filenames = np.asarray(filenames)
        if resizeSize > 0: self.transform = generateTransform([resizeSize, resizeSize], True, True)
        else: self.transform = generateTransform([], True, True)
    
    #Rearranging import dimensions, allows resize transform before tensor-conversion/rescaling; preserves precision and output data range
    def __getitem__(self, index): 
        return self.transform(np.moveaxis(cv2.cvtColor(cv2.imread(self.filenames[index], cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB), -1, 0))
    
    def __len__(self): 
        return len(self.filenames)

#Convert numpy array to contiguous tensor; issues with lambda functions when using multiprocessing
class ContiguousTensor(torch.nn.Module):
    def forward(self, data):
        return torch.from_numpy(data).contiguous()

#Generate a torch transform needed for preprocessing image data; rescale and change data type only after resize (otherwise can escape [0, 1])
def generateTransform(resizeSize=[], rescale=False, normalize=False):
    transform = [ContiguousTensor()]
    if len(resizeSize) > 0: transform.append(v2.Resize(tuple(resizeSize)))
    transform.append(v2.ToDtype(torch.get_default_dtype(), scale=rescale))
    if normalize: transform.append(v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    return v2.Compose(transform)
