#==================================================================
#RUN_CLASS
#==================================================================

#Load and process metadata
if classifierTrain or classifierExport or classifierRecon:
    
    #Attempt to load and process metadata for block images and their specific WSI
    try:
        blockSampleNames_blocks, indices_blocks, locations_blocks, blockLabels_blocks = loadMetadata_blocks(dir_blocks_inputBlocks + 'metadata_blocks.csv')
        blockLabels_blocks = blockLabels_blocks.astype(int)
        blockNames_blocks = np.asarray([blockSampleNames_blocks[index] + '_' + indices_blocks[index] for index in range(0, len(blockSampleNames_blocks))])
        blockFilenames_blocks = np.asarray([dir_blocks_inputBlocks + blockSampleNames_blocks[index] + os.path.sep + 'PS'+blockSampleNames_blocks[index]+'_'+str(indices_blocks[index])+'_'+str(locations_blocks[index, 0])+'_'+str(locations_blocks[index, 1])+'.tif' for index in range(0, len(blockSampleNames_blocks))])
        sampleNames_blocks = np.unique(blockSampleNames_blocks)
        WSIFilenames_blocks = np.asarray([dir_blocks_inputWSI + sampleName + '.jpg' for sampleName in sampleNames_blocks])
    except:
        if classifierTrain or classifierExport: sys.exit('\nError - Failed to load data needed for classifierTrain\n')
        print('\nWarning - Failed to find/load data in: ' + dir_blocks_inputBlocks + '\n')
        blockSampleNames_blocks, indices_blocks, locations_blocks, blockLabels_blocks = np.asarray([]), np.asarray([]), np.asarray([]), np.asarray([])
    
    #Load and determine sample names for all WSI (not just those needed for labeled block images)
    WSIFilenames_recon = np.asarray(natsort.natsorted(glob.glob(dir_blocks_inputWSI + '*.jpg'))+natsort.natsorted(glob.glob(dir_recon_inputWSI + '*.jpg')))
    sampleNames_recon = np.asarray([os.path.basename(filename).split('.jpg')[0] for filename in WSIFilenames_recon])

#If configured for block classification and/or model export
if classifierTrain: 
    
    #Declare section execution in UI
    sectionTitle('GENERATING AND EVALUATING BLOCK & WSI CLASSIFIER')
    
    #Create classifier model
    modelClassifier_blocks = Classifier('blocks', blockNames_blocks, blockFilenames_blocks, blockSampleNames_blocks, locations_blocks, sampleNames_blocks, WSIFilenames_blocks, blockLabels=blockLabels_blocks)
    
    #Train model components; 'original' uses cross-validation for evaluation and trains on all data when exporting
    if classifierModel == 'updated': modelClassifier_blocks.train()
    
    #Export model components
    if classifierExport: modelClassifier_blocks.exportClassifier()
    
    #Evaluate classifier model components
    if classifierEvaluate: 
        if classifierModel == 'original': modelClassifier_blocks.crossValidation()
        elif classifierModel == 'updated': modelClassifier_blocks.evaluate()
    
    #Clean RAM of larger object(s)
    del modelClassifier_blocks
    cleanup()

#If data should be generated for reconstruction model, extract blocks/data for WSI used to train the classifier, merge with those not used, and export for later use
if classifierRecon: 
    
    #Declare section execution in UI
    sectionTitle('CLASSIFYING REMAINING WSI & ASSEMBLING RECONSTRUCTOR DATA')
    
    #If a background value for thresholding block data has not been set, determine one across all available WSI
    if blockBackgroundValue == -1:
        otsuThresholds = np.asarray([cv2.threshold(cv2.cvtColor(cv2.imread(filename, cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2GRAY),0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[0] for filename in tqdm(WSIFilenames_recon, desc='Background Estimation', leave=True, ascii=asciiFlag)])
        blockBackgroundValue = int(otsuThresholds.min())
        print('Given all available WSI, the recommended value that will be used for blockBackgroundValue is: '+str(blockBackgroundValue))
    
    #Split the WSI into blocks if the overwrite is enabled (do not reuse labeled blocks; different extraction process used)
    if overwrite_recon_blocks: 
        blockNames_recon, blockFilenames_recon, blockSampleNames_recon, blockLocations_recon, cropData_recon, paddingData_recon, shapeData_recon = extractBlocks(WSIFilenames_recon, blockBackgroundValue, dir_recon_blocks)
        blockData_recon = np.concatenate([np.expand_dims(blockNames_recon, -1), np.expand_dims(blockFilenames_recon, -1), np.expand_dims(blockSampleNames_recon, -1), blockLocations_recon], 1)
        np.save(dir_recon_blocks + 'blockData_recon', blockData_recon)
        WSIData_recon = np.concatenate([cropData_recon, paddingData_recon, shapeData_recon], 1)
        np.save(dir_recon_blocks + 'WSIData_recon', WSIData_recon)
    else: 
        blockData_recon = np.load(dir_recon_blocks + 'blockData_recon.npy', allow_pickle=True)
        blockNames_recon, blockFilenames_recon, blockSampleNames_recon, blockLocations_recon = [np.squeeze(data) for data in np.split(blockData_recon, [1, 2, 3], 1)]
        blockLocations_recon = blockLocations_recon.astype(int)
        WSIData_recon = np.load(dir_recon_blocks + 'WSIData_recon.npy', allow_pickle=True)
        cropData_recon, paddingData_recon, shapeData_recon = [np.squeeze(data) for data in np.split(WSIData_recon, [4, 8], 1)]
    
    #Prepare classifier; loading the pre-trained model
    modelClassifier_recon = Classifier('recon', blockNames_recon, blockFilenames_recon, blockSampleNames_recon, blockLocations_recon, sampleNames_recon, WSIFilenames_recon, cropData=cropData_recon, paddingData=paddingData_recon, shapeData=shapeData_recon, suffix='_recon')
    modelClassifier_recon.loadClassifier()
    
    #Classify all WSI to generate data for the reconstruction model
    modelClassifier_recon.generateReconData()
    
    #Clean RAM of larger object(s)
    del modelClassifier_recon
    cleanup()