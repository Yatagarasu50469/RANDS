#==================================================================
#RUN_CLASS
#==================================================================

#Load and process metadata
if classifierTrain or classifierExport or classifierRecon:
    
    #Attempt to load and process metadata for patch images and their specific WSI
    try:
        patchSampleNames_patches, indices_patches, locations_patches, patchLabels_patches = loadMetadata_patches(dir_patches_inputPatches + 'metadata_patches.csv')
        patchLabels_patches = patchLabels_patches.astype(int)
        patchNames_patches = np.asarray([patchSampleNames_patches[index] + '_' + indices_patches[index] for index in range(0, len(patchSampleNames_patches))])
        patchFilenames_patches = np.asarray([dir_patches_inputPatches + patchSampleNames_patches[index] + os.path.sep + 'PS'+patchSampleNames_patches[index]+'_'+str(indices_patches[index])+'_'+str(locations_patches[index, 0])+'_'+str(locations_patches[index, 1])+'.tif' for index in range(0, len(patchSampleNames_patches))])
        WSISampleNames_WSI = np.asarray(natsort.natsorted(np.unique(patchSampleNames_patches)))
        extWSI = list(np.unique([os.path.basename(filename).split('.')[1] for filename in glob.glob(dir_patches_inputWSI+'*')]))
        if 'csv' in extWSI: extWSI.remove('csv')
        if len(extWSI) > 1: sys.exit('ERROR - Multiple extensions seen for images in: ', dir_patches_inputWSI)
        else: extWSI = '.'+extWSI[0]
        WSIFilenames_patches = np.asarray([dir_patches_inputWSI + sampleName + extWSI for sampleName in WSISampleNames_WSI])
    except:
        if classifierTrain or classifierExport: sys.exit('\nError - Failed to load patch data needed for classifierTrain\n')
        print('\nWarning - Failed to find/load data in: ' + dir_patches_inputPatches + '\n')
        patchSampleNames_patches, indices_patches, locations_patches, patchLabels_patches = np.asarray([]), np.asarray([]), np.asarray([]), np.asarray([])
    
    #Determmine WSI-level labels based on if any patch-level label is malignant
    WSILabels_WSI = np.asarray([valueMalignant if valueMalignant in patchLabels_patches[patchSampleNames_patches==sampleName] else valueBenign for sampleName in WSISampleNames_WSI])
    
    #Load and determine sample names for all WSI (not just those needed for labeled patch images)
    WSIFilenames_recon = np.asarray(natsort.natsorted(glob.glob(dir_patches_inputWSI + '*'+extWSI))+natsort.natsorted(glob.glob(dir_recon_inputWSI + '*'+extWSI)))
    sampleNames_recon = np.asarray([os.path.basename(filename).split(extWSI)[0] for filename in WSIFilenames_recon])

#If configured for patch classification and/or model export
if classifierTrain: 
    
    #Declare section execution in UI
    sectionTitle('GENERATING AND EVALUATING PATCH & WSI CLASSIFIER')
    
    #Create classifier model
    modelClassifier_patches = Classifier('patches', patchNames_patches, patchFilenames_patches, patchSampleNames_patches, locations_patches, WSISampleNames_WSI, WSIFilenames_patches, WSILabels=WSILabels_WSI, patchLabels=patchLabels_patches)
    
    #Train model components; 'original' uses cross-validation for evaluation and trains on all data when exporting
    if classifierModel == 'updated': modelClassifier_patches.train()
    
    #Export model components
    if classifierExport: modelClassifier_patches.exportClassifier()
    
    #Evaluate classifier model components
    if classifierEvaluate: 
        if classifierModel == 'original': modelClassifier_patches.crossValidation()
        elif classifierModel == 'updated': modelClassifier_patches.evaluate()
    
    #Clean RAM of larger object(s)
    del modelClassifier_patches
    cleanup()

#If data should be generated for reconstruction model, extract patches/data for WSI used to train the classifier, merge with those not used, and export for later use
if classifierRecon: 
    
    #Declare section execution in UI
    sectionTitle('CLASSIFYING REMAINING WSI & ASSEMBLING RECONSTRUCTOR DATA')
    
    #Split the WSI into patches if the overwrite is enabled (do not reuse labeled patches; different extraction process used)
    if overwrite_recon_patches: 
        patchNames_recon, patchFilenames_recon, patchSampleNames_recon, patchLocations_recon, cropData_recon, paddingData_recon, shapeData_recon = extractPatchesMultiple(WSIFilenames_recon, patchBackgroundValue, dir_recon_patches)
        patchData_recon = np.concatenate([np.expand_dims(patchNames_recon, -1), np.expand_dims(patchFilenames_recon, -1), np.expand_dims(patchSampleNames_recon, -1), patchLocations_recon], 1)
        np.save(dir_recon_patches + 'patchData_recon', patchData_recon)
        WSIData_recon = np.concatenate([cropData_recon, paddingData_recon, shapeData_recon], 1)
        np.save(dir_recon_patches + 'WSIData_recon', WSIData_recon)
    else: 
        patchData_recon = np.load(dir_recon_patches + 'patchData_recon.npy', allow_pickle=True)
        patchNames_recon, patchFilenames_recon, patchSampleNames_recon, patchLocations_recon = [np.squeeze(data) for data in np.split(patchData_recon, [1, 2, 3], 1)]
        patchLocations_recon = patchLocations_recon.astype(int)
        WSIData_recon = np.load(dir_recon_patches + 'WSIData_recon.npy', allow_pickle=True)
        cropData_recon, paddingData_recon, shapeData_recon = [np.squeeze(data) for data in np.split(WSIData_recon, [4, 8], 1)]
    
    #Prepare classifier; loading the pre-trained model
    modelClassifier_recon = Classifier('recon', patchNames_recon, patchFilenames_recon, patchSampleNames_recon, patchLocations_recon, sampleNames_recon, WSIFilenames_recon, cropData=cropData_recon, paddingData=paddingData_recon, shapeData=shapeData_recon, suffix='_recon')
    modelClassifier_recon.loadClassifier()
    
    #Classify all WSI to generate data for the reconstruction model
    modelClassifier_recon.generateReconData()
    
    #Clean RAM of larger object(s)
    del modelClassifier_recon
    cleanup()