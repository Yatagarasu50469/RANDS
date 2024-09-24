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
        
        #Prepare data objects for obtaining/processing PyTorch model inputs
        #Using num_workers>1 in the DataLoader objects causes bizzare semaphore/lock/descriptor issues/warnings; still appears to work, but keeping to 1 for safety
        self.device = f"cuda:{gpus[-1]}" if len(gpus) > 0 else "cpu"
        self.torchDevice = torch.device(self.device)
        self.blockData = DataPreprocessing_Classifier(self.blockFilenames, resizeSize_blocks)
        self.blockDataloader = DataLoader(self.blockData, batch_size=batchsizeClassifier, num_workers=1, shuffle=False, pin_memory=True)
        self.numBlockData = len(self.blockDataloader)
        self.WSIData = DataPreprocessing_Classifier(self.WSIFilenames, resizeSize_WSI)
        self.WSIDataloader = DataLoader(self.WSIData, batch_size=batchsizeClassifier, num_workers=1, shuffle=False, pin_memory=True)
        self.numWSIData = len(self.WSIDataloader)
        
        #Set default cuda device for XGBClassifier input data
        if len(gpus) > 0: cp.cuda.Device(gpus[-1]).use()
        
        #Specify internal object data/directories according to data type
        if dataType == 'blocks':
            self.overwrite_features = overwrite_blocks_features
            self.overwrite_saliencyMaps = overwrite_blocks_saliencyMaps
            self.visualizeSaliencyMaps = visualizeSaliencyMaps_blocks
            self.dir_results = dir_blocks_results
            self.dir_features = dir_blocks_features
            self.dir_saliencyMaps = dir_blocks_salicencyMaps
            self.dir_visuals_saliencyMaps = dir_blocks_visuals_saliencyMaps
            self.dir_visuals_overlaidSaliencyMaps = dir_blocks_visuals_overlaidSaliencyMaps
            self.dir_visuals_labelGrids = dir_blocks_visuals_labelGrids
            self.dir_visuals_overlaidLabelGrids = dir_blocks_visuals_overlaidLabelGrids
            self.dir_results_labelGrids = dir_blocks_results_labelGrids
            self.dir_results_overlaidLabelGrids = dir_blocks_results_overlaidLabelGrids
            self.visualizeLabelGrids = visualizeLabelGrids_blocks
            self.visualizePredictionGrids = visualizePredictionGrids_blocks
        elif 'WSI' in dataType or dataType == 'recon':
            if dataType == 'WSI_WSI':
                self.overwrite_features = overwrite_WSI_features
                self.overwrite_saliencyMaps = overwrite_WSI_saliencyMaps
            elif dataType == 'WSI_blocks':
                self.overwrite_features = overwrite_recon_features
                self.overwrite_saliencyMaps = overwrite_recon_saliencyMaps
            self.dir_results = dir_WSI_results
            self.dir_features = dir_WSI_features
            self.dir_saliencyMaps = dir_WSI_saliencyMaps
            self.dir_visuals_saliencyMaps = dir_WSI_visuals_saliencyMaps
            self.dir_visuals_overlaidSaliencyMaps = dir_WSI_visuals_overlaidSaliencyMaps
            self.dir_results_labelGrids = dir_WSI_results_labelGrids
            self.dir_results_overlaidLabelGrids = dir_WSI_results_overlaidLabelGrids
            self.visualizeSaliencyMaps = visualizeSaliencyMaps_WSI
            self.visualizeLabelGrids = False
            self.visualizePredictionGrids = visualizePredictionGrids_WSI
        else:
            sys.exit('\nError - Unknown data type used when creating classifier object.')
        
        #Compute or load/combine features and saliency maps
        if (dataType == 'blocks') or ('WSI' in dataType):
            self.computeFeatures()
            if (dataType == 'blocks' and fusionMode_blocks) or ('WSI' in dataType and fusionMode_WSI): self.computeSalicencyMaps()
        elif dataType == 'recon':
            blockFeatures_WSI_blocks = np.load(self.dir_features + 'blockFeatures_WSI_blocks.npy', allow_pickle=True)
            blockFeatures_WSI_WSI = np.load(self.dir_features + 'blockFeatures_WSI_WSI.npy', allow_pickle=True)
            self.blockFeatures = np.concatenate([blockFeatures_WSI_blocks, blockFeatures_WSI_WSI])
            np.save(dir_recon_inputData + 'blockFeatures', self.blockFeatures)
            del blockFeatures_WSI_WSI, blockFeatures_WSI_blocks
            if fusionMode_WSI:
                blockWeights_WSI_blocks = np.load(self.dir_saliencyMaps + 'blockWeights_WSI_blocks.npy', allow_pickle=True)
                blockWeights_WSI_WSI = np.load(self.dir_saliencyMaps + 'blockWeights_WSI_WSI.npy', allow_pickle=True)
                self.blockWeights = np.concatenate([blockWeights_WSI_blocks, blockWeights_WSI_WSI])
                np.save(dir_recon_inputData + 'blockWeights', self.blockWeights)
                del blockWeights_WSI_WSI, blockWeights_WSI_blocks
        
    def computeFeatures(self):
        
        #Extract or load features for the indicated block files
        if self.overwrite_features: 
        
            #Load pretrained ResNet50 model and set to evaluation mode
            model_ResNet = models.resnet50(weights=weightsResNet).to(self.torchDevice)
            _ = model_ResNet.train(False)

            #Extract features for each batch of sample block images
            self.blockFeatures = []
            for data in tqdm(self.blockDataloader, total=self.numBlockData, desc='Feature Determination', leave=True, ascii=asciiFlag):
                self.blockFeatures += model_ResNet(data.to(self.torchDevice)).detach().cpu().tolist()
            
            #Clear the ResNet model
            del model_ResNet
            if len(gpus) > 0: torch.cuda.empty_cache()
            
            #Convert list of features to an array
            self.blockFeatures = np.asarray(self.blockFeatures)
            
            #Save features to disk
            np.save(self.dir_features + 'blockFeatures'+self.suffix, self.blockFeatures)
            
        else: 
            self.blockFeatures = np.load(self.dir_features + 'blockFeatures'+self.suffix+'.npy', allow_pickle=True)
        
    def computeSalicencyMaps(self):
        
        #Extract or load saliency map data for the indicated block files
        if self.overwrite_saliencyMaps:
            
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

            #Extract features for each batch of sample block images
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
            self.blockWeights = []
            for index, saliencyMap in tqdm(enumerate(self.saliencyMaps), total=len(self.saliencyMaps), desc='Saliency Processing', leave=True, ascii=asciiFlag): 
                
                #Load the sample WSI
                imageWSI = cv2.cvtColor(cv2.imread(self.WSIFilenames[index], cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB)
                
                #If visualizations are enabled
                if self.visualizeSaliencyMaps:
                    
                    #Store the saliency map to disk (keep at original output dimensions)
                    exportImage(self.dir_visuals_saliencyMaps+'saliencyMap_'+self.sampleNames[index]+'.tif', matplotlib.cm.jet(saliencyMap)[:,:,:-1].astype(np.float32))
                
                    #Resize the saliency map to match the WSI dimensions
                    transform = generateTransform(imageWSI.shape[:2], False, False)
                    saliencyMap = transform(np.expand_dims(saliencyMap, 0))[0].numpy()
                    
                    #Overlay the saliency map on the WSI and save it to disk
                    transform = generateTransform([], True, False)
                    overlaid = np.moveaxis(transform(np.moveaxis(imageWSI, -1, 0)).numpy(), 0, -1)
                    if overlayGray: overlaid = np.expand_dims(cv2.cvtColor(overlaid, cv2.COLOR_RGB2GRAY), -1)
                    overlaid = show_cam_on_image(overlaid, saliencyMap, use_rgb=True, colormap=cv2.COLORMAP_HOT, image_weight=1.0-overlayWeight)
                    exportImage(self.dir_visuals_overlaidSaliencyMaps+'overlaidSaliency_'+self.sampleNames[index]+overlayExtension, overlaid)
                
                #Extract saliency map data specific to sample block locations, compute regional importance as the average value, and threshold to get weights
                blockWeights = []
                for locationIndex, locationData in enumerate(self.blockLocations[np.where(self.blockSampleNames == self.sampleNames[index])[0]]):
                    startRow, startColumn = locationData
                    blockSaliencyMap = saliencyMap[startRow:startRow+blockSize, startColumn:startColumn+blockSize]
                    blockImportance = np.mean(blockSaliencyMap)
                    if blockImportance < 0.25: blockWeights.append(0)
                    else: blockWeights.append(blockImportance)
                self.blockWeights += blockWeights
                
            #Convert list of block weights to an array and save to disk
            self.blockWeights = np.asarray(self.blockWeights)
            np.save(self.dir_saliencyMaps + 'blockWeights'+self.suffix, self.blockWeights)
            
        else:
            self.blockWeights = np.load(self.dir_saliencyMaps + 'blockWeights'+self.suffix+'.npy', allow_pickle=True)
    
    #Classify extracted block features
    def predict(self, inputs, fusionMode, weights=None):
        
        #Compute the raw block predictions
        predictions = self.model_XGBClassifier.predict(inputs.astype(np.float32))
        
        #If fusion mode is active, multiply the block predictions (using -1 for benign and +1 for malignant) by the matching weights; positive results are malignant
        if fusionMode: 
            predictionsFusion = np.where(predictions==0, -1, 1)*weights
            predictionsFusion = np.where(predictionsFusion>0, 1, 0)
            return predictions.tolist(), predictionsFusion.tolist()
        else: 
            return predictions.tolist()
    
    #Perform cross-validation
    def crossValidation(self):
        
        #If block weights are available, then enable evaluation of fusion mode
        if len(self.blockWeights)>0: fusionMode = True
        else: fusionMode = False
        
        #Allocate samples to folds for cross validation
        if type(manualFolds) != list: folds = [array.tolist() for array in np.array_split(np.random.permutation(self.sampleNames), manualFolds)]
        else: folds = manualFolds
        numFolds = len(folds)

        #Split block features into specified folds, keeping track of originating indices and matched labels
        foldsFeatures, foldsLabels, foldsWeights, foldsBlockSampleNames, foldsBlockNames, foldsBlockLocations, foldsWSILabels = [], [], [], [], [], [], []
        for fold in folds:
            blockIndices = np.concatenate([np.where(self.blockSampleNames == sampleName)[0] for sampleName in fold])
            foldsBlockLocations.append(list(self.blockLocations[blockIndices]))
            foldsFeatures.append(list(self.blockFeatures[blockIndices]))
            foldsLabels.append(list(self.blockLabels[blockIndices]))
            if fusionMode: foldsWeights.append(list(self.blockWeights[blockIndices]))
            foldsBlockSampleNames.append(list(self.blockSampleNames[blockIndices]))
            foldsBlockNames.append(list(self.blockNames[blockIndices]))            
            foldsWSILabels += [self.WSILabels[np.where(self.sampleNames == sampleName)[0]][0] for sampleName in fold]
        
        #Collapse data for later (correct/matched ordered) evaluation of the fold data
        foldsSampleNames = np.asarray(sum(folds, []))
        foldsBlockLocations = np.concatenate(foldsBlockLocations)
        foldsBlockSampleNames = np.concatenate(foldsBlockSampleNames)
        foldsBlockNames = np.concatenate(foldsBlockNames)
        foldsWSILabels = np.asarray(foldsWSILabels)
        foldsWSIFilenames = np.concatenate([self.WSIFilenames[np.where(self.sampleNames == sampleName)[0]] for sampleName in foldsSampleNames])
        
        #Check class distribution between the folds
        #print('B\t M \t Total')
        #for foldNum in range(0, numFolds):
        #    print(np.sum(np.array(foldsLabels[foldNum]) == valueBenign),'\t', np.sum(np.array(foldsLabels[foldNum]) == valueMalignant), '\t', len(foldsLabels[foldNum]))
        
        #Perform training/testing among the folds, testing sequentially (to ensure the correct order) and storing results for later evaluation
        foldsBlockPredictions, foldsBlockPredictionsFusion = [], []
        for foldNum in tqdm(range(0, numFolds), desc='Block Classification', leave=True, ascii=asciiFlag):
            
            #Train on all folds except the one specified
            self.train(np.concatenate(foldsFeatures[:foldNum]+foldsFeatures[foldNum+1:]), np.concatenate(foldsLabels[:foldNum]+foldsLabels[foldNum+1:]))
            
            #Extract the testing data and place on the GPU if able
            dataInput = np.asarray(foldsFeatures[foldNum])
            if len(gpus) > 0: dataInput = cp.asarray(dataInput)
            
            #Classify blocks in the specified, remaining fold
            if fusionMode: 
                predictions, predictionsFusion = self.predict(dataInput, fusionMode, np.asarray(foldsWeights[foldNum]))
                foldsBlockPredictionsFusion += predictionsFusion
            else:
                predictions = self.predict(dataInput, fusionMode)
            foldsBlockPredictions += predictions
            
            #Clear the XGBClassifier model and data on GPU
            del self.model_XGBClassifier, dataInput
            if len(gpus) > 0: 
                torch.cuda.empty_cache() 
                cp._default_memory_pool.free_all_blocks()
        
        #Convert lists of predictions to arrays and collapse labels for evaluation
        foldsLabels = np.concatenate(foldsLabels)
        foldsBlockPredictions = np.asarray(foldsBlockPredictions)
        foldsBlockPredictionsFusion = np.asarray(foldsBlockPredictionsFusion)
        
        #Save results to disk
        dataPrintout, dataPrintoutNames = [foldsBlockNames, foldsLabels, foldsBlockPredictions], ['Names', 'Labels', 'Raw Predictions']
        if fusionMode: 
            dataPrintout.append(foldsBlockPredictionsFusion)
            dataPrintoutNames.append('Fusion Predictions')
        dataPrintout = pd.DataFrame(np.asarray(dataPrintout)).transpose()
        dataPrintout.columns=dataPrintoutNames
        dataPrintout.to_csv(self.dir_results + 'predictions_blocks.csv', index=False)
        
        #Evaluate per-block results
        computeClassificationMetrics(foldsLabels, foldsBlockPredictions, self.dir_results, '_blocks_initial')
        if fusionMode: computeClassificationMetrics(foldsLabels, foldsBlockPredictionsFusion, self.dir_results, '_blocks_fusion')
    
        #Classify each WSI, that had its component blocks classified, according to specified threshold of allowable malignant blocks
        foldsSampleLabels, foldsSamplePredictions, foldsSamplePredictionsFusion = [], [], []
        for foldsSampleIndex, sampleName in enumerate(foldsSampleNames):
            blockIndices = np.where(foldsBlockSampleNames == sampleName)[0]
            if thresholdWSI_GT: foldsSampleLabels.append((np.mean(foldsLabels[blockIndices]) >= thresholdWSI)*1)
            foldsSamplePredictions.append((np.mean(foldsBlockPredictions[blockIndices]) >= thresholdWSI)*1)
            if fusionMode: foldsSamplePredictionsFusion.append((np.mean(foldsBlockPredictionsFusion[blockIndices]) >= thresholdWSI)*1)
        
        #Convert list of WSI labels/predictions to arrays
        if thresholdWSI_GT: 
            foldsSampleLabels = np.asarray(foldsSampleLabels)
            mismatchedSamples = foldsSampleNames[np.where(foldsSampleLabels-foldsWSILabels != 0)[0]].tolist()
            if len(mismatchedSamples) > 0: print('\nWarning - Determination of ground-truth labels for WSI using threshold method did not match with WSI labels in recorded metadata for the following samples: ' + str(mismatchedSamples))
        else: foldsSampleLabels = foldsWSILabels
        foldsSamplePredictions = np.asarray(foldsSamplePredictions)
        foldsSamplePredictionsFusion = np.asarray(foldsSamplePredictionsFusion)
        
        #Evaluate per-sample results
        self.processResultsWSI(foldsSampleNames, foldsWSIFilenames, foldsSampleLabels, foldsSamplePredictions, foldsSamplePredictionsFusion, foldsBlockSampleNames, foldsLabels, foldsBlockPredictions, foldsBlockPredictionsFusion, foldsBlockLocations)

    def processResultsWSI(self, sampleNames, WSIFilenames, sampleLabels, samplePredictions, samplePredictionsFusion, blockSampleNames, blockLabels, blockPredictions, blockPredictionsFusion, blockLocations):
        
        #If block weights are available, then enable evaluation of fusion mode
        if len(self.blockWeights)>0: fusionMode = True
        else: fusionMode = False
        
        #Save results to disk
        dataPrintout, dataPrintoutNames = [sampleNames, sampleLabels, samplePredictions], ['Names', 'Labels', 'Raw Predictions']
        if fusionMode: 
            dataPrintout.append(samplePredictionsFusion)
            dataPrintoutNames.append('Fusion Predictions') 
        dataPrintout = pd.DataFrame(np.asarray(dataPrintout)).transpose()
        dataPrintout.columns=dataPrintoutNames
        dataPrintout.to_csv(self.dir_results + 'predictions_WSI.csv', index=False)
        
        #Evaluate WSI results
        computeClassificationMetrics(sampleLabels, samplePredictions, self.dir_results, '_WSI_initial')
        if fusionMode: computeClassificationMetrics(sampleLabels, samplePredictionsFusion, self.dir_results, '_WSI_fusion')
        
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
                    if fusionMode: 
                        gridOverlay_PredictionsFusion = np.zeros(imageWSI.shape, dtype=np.uint8)
                        colorsFusion = (cmapClasses(blockPredictionsFusion)[:,:3].astype(np.uint8)*255).tolist()
                if self.visualizeLabelGrids: 
                    gridOverlay_GT = np.zeros(imageWSI.shape, dtype=np.uint8)
                    colorsLabels = (cmapClasses(blockLabels)[:,:3].astype(np.uint8)*255).tolist()
                for blockIndex in blockIndices:
                    startRow, startColumn = blockLocations[blockIndex] 
                    posStart, posEnd = (startColumn, startRow), (startColumn+blockSize, startRow+blockSize)
                    if self.visualizeLabelGrids: gridOverlay_GT = rectangle(gridOverlay_GT, posStart, posEnd, colorsLabels[blockIndex])
                    if self.visualizePredictionGrids:
                        gridOverlay_Predictions = rectangle(gridOverlay_Predictions, posStart, posEnd, colorsPredictions[blockIndex])
                        if fusionMode: gridOverlay_PredictionsFusion = rectangle(gridOverlay_PredictionsFusion, posStart, posEnd, colorsFusion[blockIndex])
                    
                #Store overlays to disk
                if self.visualizeLabelGrids: exportImage(self.dir_visuals_labelGrids+'overlay_labelGrid_'+sampleName+overlayExtension, gridOverlay_GT)
                if self.visualizePredictionGrids:
                    exportImage(self.dir_results_labelGrids+'overlay_predictionsGrid_'+sampleName+overlayExtension, gridOverlay_Predictions)
                    if fusionMode: exportImage(self.dir_results_labelGrids+'overlay_fusionGrid_'+sampleName+overlayExtension, gridOverlay_PredictionsFusion)
                
                #Overlay grids on top of WSI and store to disk
                if self.visualizeLabelGrids: 
                    imageWSI_GT = cv2.addWeighted(imageWSI, 1.0, gridOverlay_GT, overlayWeight, 0.0)
                    exportImage(self.dir_visuals_overlaidLabelGrids+'labelGrid_'+sampleName+overlayExtension, imageWSI_GT)
                if self.visualizePredictionGrids: 
                    imageWSI_Predictions = cv2.addWeighted(imageWSI, 1.0, gridOverlay_Predictions, overlayWeight, 0.0)
                    exportImage(self.dir_results_overlaidLabelGrids+'predictionsGrid_'+sampleName+overlayExtension, imageWSI_Predictions)
                    if fusionMode: 
                        imageWSI_PredictionsFusion = cv2.addWeighted(imageWSI, 1.0, gridOverlay_PredictionsFusion, overlayWeight, 0.0)
                        exportImage(self.dir_results_overlaidLabelGrids+'fusionGrid_'+sampleName+overlayExtension, imageWSI_PredictionsFusion)
    
    #Train a new XGB Classifier model
    def train(self, inputs, labels):
        self.model_XGBClassifier = XGBClassifier(device=self.device)
        _  = self.model_XGBClassifier.fit(inputs.astype(np.float32), labels)
    
    #Train on all available data and export models
    def exportClassifier(self):
        
        #Setup, train, save, and clear the XGBClassifier model
        self.train(self.blockFeatures, self.blockLabels)
        
        #Save to disk in .json format for easy reloading
        self.model_XGBClassifier.save_model(dir_classifier_models + 'model_XGBClassifier.json')
        
        #Register converter for XGBClassifier
        update_registered_converter(XGBClassifier, "XGBoostXGBClassifier", calculate_linear_classifier_output_shapes, convert_xgboost, options={"nocl": [True, False], "zipmap": [True, False, "columns"]},)
        
        #Convert classifier to onnx format and save to disk
        model_onnx_XGBClassifier = to_onnx(self.model_XGBClassifier, self.blockFeatures.astype(np.float32), target_opset={"": skl2onnx.__max_supported_opset__, "ai.onnx.ml": 3})
        with open(dir_classifier_models + 'model_XGBClassifier.onnx', 'wb') as f:f.write(model_onnx_XGBClassifier.SerializeToString())
        
        #Clear memory
        del self.model_XGBClassifier, model_onnx_XGBClassifier
        if len(gpus) > 0: 
            torch.cuda.empty_cache() 
            cp._default_memory_pool.free_all_blocks()
        
        #Setup pre-trained ResNet model
        #model_ResNet = models.resnet50(weights=weightsResNet)
        
        #Convert to onnx format and save to disk
        #model_onnx_ResNet = 
        #Tracer()
        #with open(dir_classifier_models + 'model_ResNet.onnx', 'wb') as f:f.write(model_onnx_ResNet.SerializeToString())
        
        #del self.model_ResNet#, model_onnx_ResNet
        #if len(gpus) > 0: 
        #    torch.cuda.empty_cache() 
        #    cp._default_memory_pool.free_all_blocks()
        
    #Load a pretrained model
    def loadClassifier(self):
        self.model_XGBClassifier = XGBClassifier()
        self.model_XGBClassifier.load_model(dir_classifier_models + 'model_XGBClassifier.json')
        self.model_XGBClassifier._Booster.set_param({'device': self.device})
        
        #Using ONNX; can be done, but GPU handling here would need work to perform correctly/well in Python
        #self.model_XGBClassifier = onnxruntime.InferenceSession(dir_classifier_models + 'model_XGBClassifier.onnx', providers = [('CUDAExecutionProvider', {"device_id": gpus[-1]}), 'CPUExecutionProvider'])
        #if len(gpus) > 0: self.model_XGBClassifier.predict = lambda inputs: session.run(None, {'X': cp.asarray(inputs)})[0]
        #else: self.model_XGBClassifier.predict = lambda inputs: session.run(None, {'X': inputs})[0]
        
    #Classify a sample WSI
    def classifyWSI(self, evaluatePredictions):
        
        #Place data on the GPU if able
        dataInput = np.asarray(self.blockFeatures.astype(np.float32))
        if len(gpus) > 0: dataInput = cp.asarray(dataInput)
        
        #Classify blocks
        if fusionMode_WSI: blockPredictions, blockPredictionsFusion = self.predict(dataInput, fusionMode_WSI, self.blockWeights)
        else: blockPredictions, blockPredictionsFusion = self.predict(dataInput, fusionMode_WSI), []
        blockPredictions, blockPredictionsFusion = np.asarray(blockPredictions), np.asarray(blockPredictionsFusion)
        
        #Clear the XGBClassifier model and data on GPU
        del self.model_XGBClassifier, dataInput
        if len(gpus) > 0: 
            torch.cuda.empty_cache() 
            cp._default_memory_pool.free_all_blocks()
        
        #Classify each WSI, that had its component blocks classified, according to specified threshold of allowable malignant blocks
        sampleLabels, samplePredictions, samplePredictionsFusion, sampleBlockIndices = [], [], [], []
        for sampleIndex, sampleName in enumerate(self.sampleNames):
            blockIndices = np.where(self.blockSampleNames == sampleName)[0]
            sampleBlockIndices.append(blockIndices)
            samplePredictions.append((np.mean(blockPredictions[blockIndices]) >= thresholdWSI)*1)
            if fusionMode_WSI: samplePredictionsFusion.append((np.mean(blockPredictionsFusion[blockIndices]) >= thresholdWSI)*1)
        
        #Convert list of WSI labels/predictions to arrays
        samplePredictions = np.asarray(samplePredictions)
        samplePredictionsFusion = np.asarray(samplePredictionsFusion)
        
        #Evaluate per-sample results
        self.processResultsWSI(self.sampleNames, self.WSIFilenames, self.WSILabels, samplePredictions, samplePredictionsFusion, self.blockSampleNames, [], blockPredictions, blockPredictionsFusion, self.blockLocations)
        
        #Create prediction arrays for reconstruction model and save them to disk
        if classifierRecon:
            predictionMaps, predictionsFusionMaps = [], []
            for sampleIndex, sampleName in tqdm(enumerate(self.sampleNames), total=len(self.sampleNames), desc='Data Assembly', leave=True, ascii=asciiFlag):
                blockIndices = sampleBlockIndices[sampleIndex]
                predictionLocations = self.blockLocations[blockIndices]//blockSize
                predictionMap = np.full((self.shapeData[sampleIndex]), valueBackground)
                predictionMap[predictionLocations[:,0], predictionLocations[:,1]] = blockPredictions[blockIndices]
                predictionMaps.append(predictionMap)
                if visualizeInputData_recon: exportImage(dir_recon_visuals_inputData+'predictionMap_'+sampleName+'.tif', cmapClasses(predictionMap)[:,:,:3].astype(np.uint8)*255)
                if fusionMode_WSI:
                    predictionFusionMap = np.full((self.shapeData[sampleIndex]), valueBackground)
                    predictionFusionMap[predictionLocations[:,0], predictionLocations[:,1]] = blockPredictionsFusion[blockIndices]
                    predictionsFusionMaps.append(predictionFusionMap)
                    if visualizeInputData_recon: exportImage(dir_recon_visuals_inputData+'fusionMap_'+sampleName+'.tif', cmapClasses(predictionFusionMap)[:,:,:3].astype(np.uint8)*255)
            predictionMaps = np.asarray(predictionMaps, dtype='object')
            np.save(dir_recon_inputData + 'predictionMaps', predictionMaps)
            if fusionMode_WSI: 
                predictionsFusionMaps = np.asarray(predictionsFusionMaps, dtype='object')
                np.save(dir_recon_inputData + 'fusionMaps', predictionsFusionMaps)
                