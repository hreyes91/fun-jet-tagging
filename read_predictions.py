
import numpy as np






def GetPredictions(predictions_path):

    #predicitions_test = np.load(classifier_dir+'/predictions_test_'+str(num_const)+'.npz')
    
    predicitions_test = np.load(predictions_path)
    
    return predicitions_test
    
        


predictions_path='/Users/humbertosmac/Dropbox/Transformers/OptimalClassifier/Data/SameBin/OptimalClassifierSamplesSameBinGen/ClassificationResults/LLRpredictions/predictions_LLR_10M_correct_128.npz'

predicitions_test = GetPredictions(predictions_path)

print(predicitions_test['labels'][-10:])
