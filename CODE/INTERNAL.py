#==================================================================
#INTERNAL
#==================================================================

#Data input directories and file locations
#=============================================================================

#Global
dir_data = '.' + os.path.sep + 'DATA' + os.path.sep
dir_results = '.' + os.path.sep + 'RESULTS' + os.path.sep
dir_classifier_models = dir_results + 'MODELS' + os.path.sep

#Patch classification
dir_patches_data = dir_data + 'PATCHES' + os.path.sep
dir_patches_inputPatches = dir_patches_data + 'INPUT_PATCHES' + os.path.sep
dir_patches_inputWSI = dir_patches_data + 'INPUT_WSI' + os.path.sep

dir_patches_features = dir_patches_data + 'OUTPUT_FEATURES' + os.path.sep
dir_patches_salicencyMaps = dir_patches_data + 'OUTPUT_SALIENCY_MAPS' + os.path.sep

dir_patches_visuals = dir_patches_data + 'OUTPUT_VISUALS' + os.path.sep
dir_patches_visuals_saliencyMaps = dir_patches_visuals + 'SALIENCY_MAPS' + os.path.sep
dir_patches_visuals_overlaidSaliencyMaps = dir_patches_visuals + 'OVERLAID_SALICENCY_MAPS' + os.path.sep
dir_patches_visuals_labelGrids = dir_patches_visuals + 'LABEL_GRIDS' + os.path.sep
dir_patches_visuals_overlaidLabelGrids = dir_patches_visuals + 'OVERLAID_LABEL_GRIDS' + os.path.sep

dir_patches_results = dir_results + 'PATCHES' + os.path.sep
dir_patches_results_visuals = dir_patches_results + 'VISUALS' + os.path.sep
dir_patches_visuals_predictionGrids = dir_patches_results_visuals + 'PREDICTION_GRIDS' + os.path.sep
dir_patches_visuals_overlaidPredictionGrids = dir_patches_results_visuals + 'OVERLAID_PREDICTION_GRIDS' + os.path.sep
dir_patches_visuals_fusionGrids = dir_patches_results_visuals + 'FUSION_GRIDS' + os.path.sep
dir_patches_visuals_overlaidFusionGrids = dir_patches_results_visuals + 'OVERLAID_FUSION_GRIDS' + os.path.sep

#Reconstruction
dir_recon_data = dir_data + 'RECON' + os.path.sep
dir_recon_inputWSI = dir_recon_data + 'INPUT_WSI' + os.path.sep
dir_recon_patches = dir_recon_data + 'OUTPUT_PATCHES' + os.path.sep
dir_recon_features = dir_recon_data + 'OUTPUT_FEATURES' + os.path.sep
dir_recon_saliencyMaps = dir_recon_data + 'OUTPUT_SALIENCY_MAPS' + os.path.sep
dir_recon_inputData = dir_recon_data + 'OUTPUT_INPUT_DATA' + os.path.sep

dir_recon_visuals = dir_recon_data + 'OUTPUT_VISUALS' + os.path.sep
dir_recon_visuals_inputData = dir_recon_visuals + 'INPUT_DATA' + os.path.sep
dir_recon_visuals_saliencyMaps = dir_recon_visuals + 'SALIENCY_MAPS' + os.path.sep
dir_recon_visuals_overlaidSaliencyMaps = dir_recon_visuals + 'OVERLAID_SALIENCY_MAPS' + os.path.sep
dir_recon_visuals_predictionGrids = dir_recon_visuals + 'PREDICTION_GRIDS' + os.path.sep
dir_recon_visuals_overlaidPredictionGrids = dir_recon_visuals + 'OVERLAID_PREDICTION_GRIDS' + os.path.sep
dir_recon_visuals_fusionGrids = dir_recon_visuals + 'FUSION_GRIDS' + os.path.sep
dir_recon_visuals_overlaidFusionGrids = dir_recon_visuals + 'OVERLAID_FUSION_GRIDS' + os.path.sep

dir_recon_results = dir_results + 'RECON' + os.path.sep
dir_recon_results_train = dir_recon_results + 'TRAIN' + os.path.sep
dir_recon_results_test = dir_recon_results + 'TEST' + os.path.sep

#Refresh/setup internal directories only in the main executing thread
if __name__ == '__main__': 

    #Define an ultimate final location for the results
    #=============================================================================

    #Indicate and setup the destination folder for results of this configuration
    dir_results_final = './RESULTS_'+os.path.splitext(os.path.basename(configFileName).split('_')[1])[0]

    #If the folder already exists, prevent name collisions by appending a novel value to it
    if os.path.exists(dir_results_final):
        destinationNameValue = 0
        dir_results_final_base = copy.deepcopy(dir_results_final)
        while True:
            dir_results_final = dir_results_final_base + '_' + str(destinationNameValue)
            if not os.path.exists(dir_results_final): break
            destinationNameValue += 1

    #If folders do not exist, but their use is enabled, exit the program
    #=============================================================================

    #Patch classification
    if not os.path.exists(dir_data): sys.exit('\nError - Required folder: ' + dir_data + ' does not exist.')
    if not os.path.exists(dir_patches_data) and (classifierTrain or classifierExport): sys.exit('\nError - Required folder: ' + dir_patches_data + ' does not exist.\n')
    if not os.path.exists(dir_patches_inputPatches) and (classifierTrain or classifierExport): sys.exit('\nError - Required folder: ' + dir_patches_inputPatches + ' does not exist.\n')
    if not os.path.exists(dir_patches_inputWSI) and (classifierTrain or classifierExport): sys.exit('\nError - Required folder: ' + dir_patches_inputWSI + ' does not exist.\n')

    #Reconstruction
    if not os.path.exists(dir_recon_data) and classifierRecon: sys.exit('\nError - Required folder: ' + dir_recon_data + ' does not exist.\n')
    if not os.path.exists(dir_recon_inputWSI) and classifierRecon: sys.exit('\nError - Required folder: ' + dir_recon_inputWSI + ' does not exist.\n')

    #If a task is diabled, then overwrites should be disabled to prevent overwrite of existing data
    #If for a given task, file overwrites are not enabled, and the needed files do not exist, then enable the relevant overwrite(s)
    #=============================================================================

    if (classifierTrain or classifierExport):
        if not overwrite_patches_features and len(glob.glob(dir_patches_features+'*.npy'))==0: overwrite_patches_features = True    

    if classifierTrain:
        if not overwrite_patches_saliencyMaps and fusionMode_patches and len(glob.glob(dir_patches_salicencyMaps+'*.npy'))==0: overwrite_patches_saliencyMaps = True
    else:
        overwrite_patches_features = False
        overwrite_patches_saliencyMaps = False

    if classifierRecon:
        if not overwrite_recon_patches and len(glob.glob(dir_recon_patches+'*.npy'))==0: overwrite_recon_patches = True
        if not overwrite_recon_features and len(glob.glob(dir_recon_features+'*.npy'))==0: overwrite_reocn_features = True
        if not overwrite_recon_saliencyMaps and fusionMode_recon and len(glob.glob(dir_recon_saliencyMaps+'*.npy'))==0: overwrite_recon_saliencyMaps = True
    else:
        overwrite_recon_patches = False
        overwrite_recon_features = False
        overwrite_recon_saliencyMaps = False

    #Clear files/folders that are to be overwritten or impacted
    #=============================================================================

    #Patch classification
    if classifierTrain: 
        if os.path.exists(dir_patches_results): shutil.rmtree(dir_patches_results)
        if classifierExport and os.path.exists(dir_classifier_models): shutil.rmtree(dir_classifier_models)
        if overwrite_patches_features and os.path.exists(dir_patches_features): shutil.rmtree(dir_patches_features)
        if overwrite_patches_saliencyMaps: 
            if os.path.exists(dir_patches_salicencyMaps): shutil.rmtree(dir_patches_salicencyMaps)
            if os.path.exists(dir_patches_visuals_saliencyMaps): shutil.rmtree(dir_patches_visuals_saliencyMaps)
            if os.path.exists(dir_patches_visuals_overlaidSaliencyMaps): shutil.rmtree(dir_patches_visuals_overlaidSaliencyMaps)
        
        if os.path.exists(dir_patches_visuals_labelGrids): shutil.rmtree(dir_patches_visuals_labelGrids)
        if os.path.exists(dir_patches_visuals_overlaidLabelGrids): shutil.rmtree(dir_patches_visuals_overlaidLabelGrids)
        if os.path.exists(dir_patches_visuals_predictionGrids): shutil.rmtree(dir_patches_visuals_PredictionGrids)
        if os.path.exists(dir_patches_visuals_overlaidPredictionGrids): shutil.rmtree(dir_patches_visuals_overlaidPredictionGrids)
        if os.path.exists(dir_patches_visuals_fusionGrids): shutil.rmtree(dir_patches_visuals_fusionGrids)
        if os.path.exists(dir_patches_visuals_overlaidFusionGrids): shutil.rmtree(dir_patches_visuals_overlaidFusionGrids)
        
    #Reconstruction
    if classifierRecon:
        if os.path.exists(dir_recon_results): shutil.rmtree(dir_recon_results)
        if os.path.exists(dir_recon_inputData): shutil.rmtree(dir_recon_inputData)
        if overwrite_recon_patches and os.path.exists(dir_recon_patches): shutil.rmtree(dir_recon_patches)
        if overwrite_recon_features and os.path.exists(dir_recon_features): shutil.rmtree(dir_recon_features)
        if overwrite_recon_saliencyMaps: 
            if os.path.exists(dir_recon_saliencyMaps): shutil.rmtree(dir_recon_saliencyMaps)
            if os.path.exists(dir_recon_visuals_saliencyMaps): shutil.rmtree(dir_recon_visuals_saliencyMaps)
            if os.path.exists(dir_recon_visuals_overlaidSaliencyMaps): shutil.rmtree(dir_recon_visuals_overlaidSaliencyMaps)
        
        if os.path.exists(dir_recon_visuals_inputData): shutil.rmtree(dir_recon_visuals_inputData)
        if os.path.exists(dir_recon_visuals_predictionGrids): shutil.rmtree(dir_recon_visuals_predictionGrids)
        if os.path.exists(dir_recon_visuals_overlaidPredictionGrids): shutil.rmtree(dir_recon_visuals_overlaidPredictionGrids)
        if os.path.exists(dir_recon_visuals_fusionGrids): shutil.rmtree(dir_recon_visuals_fusionGrids)
        if os.path.exists(dir_recon_visuals_overlaidFusionGrids): shutil.rmtree(dir_recon_visuals_overlaidFusionGrids)
        
    #Create any required folders that do not exist
    #=============================================================================

    #Store directories to check and create
    checkDirectories = [dir_results]

    #Patch classification
    checkDirectories += [dir_patches_features, 
                        dir_patches_salicencyMaps,
                        dir_patches_visuals,
                        dir_patches_visuals_saliencyMaps,
                        dir_patches_visuals_overlaidSaliencyMaps,
                        dir_patches_visuals_labelGrids, 
                        dir_patches_visuals_overlaidLabelGrids,
                        dir_classifier_models,
                        dir_patches_results,
                        dir_patches_results_visuals, 
                        dir_patches_visuals_predictionGrids, 
                        dir_patches_visuals_overlaidPredictionGrids, 
                        dir_patches_visuals_fusionGrids, 
                        dir_patches_visuals_overlaidFusionGrids
                       ]

    #Reconstruction
    checkDirectories += [dir_recon_patches, 
                        dir_recon_features,
                        dir_recon_saliencyMaps,
                        dir_recon_inputData, 
                        dir_recon_visuals, 
                        dir_recon_visuals_inputData,
                        dir_recon_visuals_saliencyMaps,
                        dir_recon_visuals_overlaidSaliencyMaps,
                        dir_recon_visuals_predictionGrids, 
                        dir_recon_visuals_overlaidPredictionGrids, 
                        dir_recon_visuals_fusionGrids, 
                        dir_recon_visuals_overlaidFusionGrids, 
                        dir_recon_results, 
                        dir_recon_results_train,
                        dir_recon_results_test
                       ]

    #Generate any missing folders
    for directory in checkDirectories: 
        if not os.path.exists(directory): os.makedirs(directory)
