#==================================================================
#RUN_CLASS
#==================================================================

#Load sample metadata for blocks and WSI data; keep seperate until forming reconstruction model input data to prevent accidental evaluation of classifier model with training data
try:
    sampleNames_blocks, WSILabels_blocks = loadMetadata_WSI(file_blocks_metadataWSI)
    WSILabels_blocks = WSILabels_blocks.astype(int)
    WSIFilenames_blocks = np.asarray([dir_blocks_inputWSI + sampleName + '.jpg' for sampleName in sampleNames_blocks])
except:
    print('\nWarning - There does not appear to be any metadata available for block samples.')
    sampleNames_blocks, WSILabels_blocks, WSIFilenames_blocks = np.asarray([]), np.asarray([]), np.asarray([])
try:
    sampleNames_WSI, WSILabels_WSI = loadMetadata_WSI(file_WSI_metadataWSI)
    WSILabels_WSI = WSILabels_WSI.astype(int)
    WSIFilenames_WSI = np.asarray([dir_WSI_inputs + sampleName + '.jpg' for sampleName in sampleNames_WSI])
except:
    print('\nWarning - There does not appear to be any metadata available for WSI samples.')
    sampleNames_WSI, WSILabels_WSI, WSIFilenames_WSI = np.asarray([]), np.asarray([]), np.asarray([])

#Combine sample names and WSI filenames for classifierRecon and/or blockBackgroundValue determination for classifierWSI
sampleNames_recon, WSIFilenames_recon = np.concatenate([sampleNames_blocks, sampleNames_WSI]), np.concatenate([WSIFilenames_blocks, WSIFilenames_WSI])

#If configured for block classification and/or model export
if classifierTrain or classifierExport: 
    
    sectionTitle('GENERATING AND EVALUATING BLOCK & WSI CLASSIFIER')
    
    #Load and process metadata for available blocks and their originating WSI
    blockSampleNames_blocks, indices_blocks, locations_blocks, blockLabels_blocks = loadMetadata_blocks(file_blocks_metadataBlocks)
    blockLabels_blocks = blockLabels_blocks.astype(int)
    blockNames_blocks = np.asarray([blockSampleNames_blocks[index] + '_' + indices_blocks[index] for index in range(0, len(blockSampleNames_blocks))])
    blockFilenames_blocks = np.asarray([dir_blocks_inputBlocks + blockSampleNames_blocks[index] + os.path.sep + 'PS'+blockSampleNames_blocks[index]+'_'+str(indices_blocks[index])+'_'+str(locations_blocks[index, 0])+'_'+str(locations_blocks[index, 1])+'.tif' for index in range(0, len(blockSampleNames_blocks))])
    
    #Create classifier model
    modelClassifier_blocks = Classifier('blocks', blockNames_blocks, blockFilenames_blocks, blockSampleNames_blocks, locations_blocks, sampleNames_blocks, WSIFilenames_blocks, WSILabels_blocks, blockLabels=blockLabels_blocks)
    
    #Train and evaluate classifier model
    if classifierModel == 'original':
        
        #Perform cross-validation
        if classifierTrain: modelClassifier_blocks.crossValidation()
        
        #Export classifier components
        if classifierExport: modelClassifier_blocks.exportClassifier()
        
        #Clean RAM of larger object(s)
        del modelClassifier_blocks
        cleanup()
    
    elif classifierModel == 'updated':
        
        #Train model
        modelClassifier_blocks.train()
        
        #Perform evaluation
        if classifierTrain: modelClassifier_blocks.evaluate()
        
        #Export classifier components
        if classifierExport: modelClassifier_blocks.exportClassifier()
        
        #Clean RAM of larger object(s)
        del modelClassifier_blocks
        cleanup()

#If classification of whole WSI or generation of data for reconstruction model should be performed
if classifierWSI or classifierRecon: 
    
    sectionTitle('CLASSIFYING & EVALUATING WSI')
    
    #If a background value for thresholding block data has not been set, determine one across all available WSI
    if blockBackgroundValue == -1:
        otsuThresholds = np.asarray([cv2.threshold(cv2.cvtColor(cv2.imread(filename, cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2GRAY),0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[0] for filename in tqdm(WSIFilenames_recon, desc='Background Estimation', leave=True, ascii=asciiFlag)])
        blockBackgroundValue = int(otsuThresholds.min())
        print('Given available WSI, the recommended value for blockBackgroundValue is: '+str(blockBackgroundValue))
    
    #If the overwrite is enabled, split WSI (excluding any that were used in training the classifier) into blocks and save to disk
    if overwrite_WSI_blocks: 
        blockNames_WSI_WSI, blockFilenames_WSI_WSI, blockSampleNames_WSI_WSI, blockLocations_WSI_WSI, cropData_WSI_WSI, paddingData_WSI_WSI, shapeData_WSI_WSI = extractBlocks(WSIFilenames_WSI, blockBackgroundValue)
        blockData_WSI = np.concatenate([np.expand_dims(blockNames_WSI_WSI, -1), np.expand_dims(blockFilenames_WSI_WSI, -1), np.expand_dims(blockSampleNames_WSI_WSI, -1), blockLocations_WSI_WSI], 1)
        np.save(dir_WSI_blocks + 'blockData_WSI_WSI', blockData_WSI)
        WSIData_WSI = np.concatenate([cropData_WSI_WSI, paddingData_WSI_WSI, shapeData_WSI_WSI], 1)
        np.save(dir_WSI_blocks + 'WSIData_WSI_WSI', WSIData_WSI)
    else: 
        blockData_WSI = np.load(dir_WSI_blocks + 'blockData_WSI_WSI.npy', allow_pickle=True)
        blockNames_WSI_WSI, blockFilenames_WSI_WSI, blockSampleNames_WSI_WSI, blockLocations_WSI_WSI = [np.squeeze(data) for data in np.split(blockData_WSI, [1, 2, 3], 1)]
        blockLocations_WSI_WSI = blockLocations_WSI_WSI.astype(int)
        WSIData_WSI = np.load(dir_WSI_blocks + 'WSIData_WSI_WSI.npy', allow_pickle=True)
        cropData_WSI_WSI, paddingData_WSI_WSI, shapeData_WSI_WSI = [np.squeeze(data) for data in np.split(WSIData_WSI, [4, 8], 1)]
        
    #Prepare classifier; loading the pre-trained model
    modelClassifier_WSI = Classifier('WSI_WSI', blockNames_WSI_WSI, blockFilenames_WSI_WSI, blockSampleNames_WSI_WSI, blockLocations_WSI_WSI, sampleNames_WSI, WSIFilenames_WSI, WSILabels_WSI, cropData=cropData_WSI_WSI, paddingData=paddingData_WSI_WSI, shapeData=shapeData_WSI_WSI, suffix='_WSI_WSI')
    modelClassifier_WSI.loadClassifier()
    
    #Classify the WSI and evaluate results
    if classifierWSI: modelClassifier_WSI.classifyWSI(True)
    
    #Clean RAM of larger object(s)
    del modelClassifier_WSI
    cleanup()
    
    #If data should be generated for reconstruction model, extract blocks/data for WSI used to train the classifier, merge with those not used, and export for later use
    if classifierRecon:
        
        sectionTitle('CLASSIFYING REMAINING WSI & ASSEMBLING RECONSTRUCTOR DATA')
        
        #Either split WSI (excluding any that were used in training the classifier) into blocks and save to disk, or load data for those previously generated 
        if overwrite_recon_blocks: 
            blockNames_WSI_blocks, blockFilenames_WSI_blocks, blockSampleNames_WSI_blocks, blockLocations_WSI_blocks, cropData_WSI_blocks, paddingData_WSI_blocks, shapeData_WSI_blocks = extractBlocks(WSIFilenames_blocks, blockBackgroundValue)
            blockData_blocks = np.concatenate([np.expand_dims(blockNames_WSI_blocks, -1), np.expand_dims(blockFilenames_WSI_blocks, -1), np.expand_dims(blockSampleNames_WSI_blocks, -1), blockLocations_WSI_blocks], 1)
            np.save(dir_WSI_blocks + 'blockData_WSI_blocks', blockData_blocks)
            WSIData_blocks = np.concatenate([cropData_WSI_blocks, paddingData_WSI_blocks, shapeData_WSI_blocks], 1)
            np.save(dir_WSI_blocks + 'WSIData_WSI_blocks', WSIData_blocks)
        else: 
            blockData_blocks = np.load(dir_WSI_blocks + 'blockData_WSI_blocks.npy', allow_pickle=True)
            blockNames_WSI_blocks, blockFilenames_WSI_blocks, blockSampleNames_WSI_blocks, blockLocations_WSI_blocks = [np.squeeze(data) for data in np.split(blockData_blocks, [1, 2, 3], 1)]
            blockLocations_WSI_blocks = blockLocations_WSI_blocks.astype(int)
            WSIData_blocks = np.load(dir_WSI_blocks + 'WSIData_WSI_blocks.npy', allow_pickle=True)
            cropData_WSI_blocks, paddingData_WSI_blocks, shapeData_WSI_blocks = [np.squeeze(data) for data in np.split(WSIData_blocks, [4, 8], 1)]
        
        #Prepare classifier
        modelClassifier_WSI_blocks = Classifier('WSI_blocks', blockNames_WSI_blocks, blockFilenames_WSI_blocks, blockSampleNames_WSI_blocks, blockLocations_WSI_blocks, sampleNames_blocks, WSIFilenames_blocks, WSILabels_blocks, cropData=cropData_WSI_blocks, paddingData=paddingData_WSI_blocks, shapeData=shapeData_WSI_blocks, suffix='_WSI_blocks')
        
        #Clean RAM of larger object(s)
        del modelClassifier_WSI_blocks
        cleanup()
        
        #Merge metadata for all WSI and store consolidated data to disk
        WSILabels_recon = np.concatenate([WSILabels_blocks, WSILabels_WSI])
        cropData_recon = np.concatenate([cropData_WSI_blocks, cropData_WSI_WSI])
        paddingData_recon = np.concatenate([paddingData_WSI_blocks, paddingData_WSI_WSI])
        shapeData_recon = np.concatenate([shapeData_WSI_blocks, shapeData_WSI_WSI])
        WSIData_recon = np.concatenate([np.expand_dims(sampleNames_recon, -1), np.expand_dims(WSIFilenames_recon, -1), np.expand_dims(WSILabels_recon, -1), cropData_recon, paddingData_recon, shapeData_recon], 1)
        np.save(dir_recon_inputData + 'WSIData_recon', WSIData_blocks)
        
        #Merge metadata for all blocks and store consolidated data to disk
        blockNames_recon = np.concatenate([blockNames_WSI_blocks, blockNames_WSI_WSI])
        blockFilenames_recon = np.concatenate([blockFilenames_WSI_blocks, blockFilenames_WSI_WSI])
        blockSampleNames_recon = np.concatenate([blockSampleNames_WSI_blocks, blockSampleNames_WSI_WSI])
        blockLocations_recon = np.concatenate([blockLocations_WSI_blocks, blockLocations_WSI_WSI], dtype=np.int64) 
        blockData_recon = np.concatenate([np.atleast_2d(blockNames_recon.T).T, np.atleast_2d(blockFilenames_recon.T).T,  np.atleast_2d(blockSampleNames_recon.T).T, blockLocations_recon], 1)
        np.save(dir_recon_inputData + 'blockData_recon', blockData_blocks)
        
        #Prepare classifier, loading the pre-trained model
        modelClassifier_recon = Classifier('recon', blockNames_recon, blockFilenames_recon, blockSampleNames_recon, blockLocations_recon, sampleNames_recon, WSIFilenames_recon, WSILabels_recon, cropData_recon, paddingData_recon, shapeData_recon)
        modelClassifier_recon.loadClassifier()
        
        #Classify the WSI, but do not evaluate as they include WSI data used in training the classifier in the first place
        modelClassifier_recon.classifyWSI(False)
        
        #Clean RAM of larger object(s)
        del modelClassifier_recon
        cleanup()