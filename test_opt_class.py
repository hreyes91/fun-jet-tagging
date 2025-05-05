import os



def read_file(file_name):

    f = open(file_name, "r")
    lines = f.readlines()

    return lines


def extract_value(var,lines):

    for line in lines:
        if 'lr_' in line:
            continue
        if var in line:
            line=line.replace(' ','')
            line=line.replace('\n','')

            value=line.split(var)[-1]
    
    return value

main_path='/net/data_ttk/hreyes/JetClass/OptClass/OptimalClassifierSamplesSameBinGen/'


data_path_2='/net/data_ttk/hreyes/JetClass/OptClass/OptimalClassifierSamplesSameBinGen/top/discretized/samples_samples_noseed_nsamples200000_trunc_5000_0.h5'


data_path_1='/net/data_ttk/hreyes/JetClass/OptClass/OptimalClassifierSamplesSameBinGen/qcd/discretized/samples_samples_noseed_nsamples200000_trunc_5000_0.h5'


main_models_path=main_path+'/BaselineTransformer/NCONST/nconst_fixedforall_2/'


models_paths=os.listdir(main_models_path)


for model_path in models_paths:


    full_path=main_models_path+'/'+model_path
    print(full_path)
    if os.path.isdir(full_path):
    
        lines=read_file(full_path+'/arguments.txt')
        num_const=extract_value('num_const',lines)
        

        command='python test_classifier.py --data_path_1 '+data_path_1+' --data_path_2 '+data_path_2+' --model_dir '+full_path+' --num_events 200000 --num_const '+str(num_const)+' --pred_name predictions_test_ordered_'+str(num_const)+'.npz'
    
    
        os.system(command)
    else:
        continue
