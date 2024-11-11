import os



#list_of_jets=['TTBar','ZJetsToNuNu','HToBB','HToCC','HToGG','HToWW2Q1L','HToWW4Q','TTBarLep','WToQ','ZToQQ']
list_of_jets=['TTBar','ZJetsToNuNu']
types_list=['train','test','val']


for jet in list_of_jets:
    for ttype in types_list:

        command='python preprocess_jetclass_onebinner.py --input_file   /net/data_t2k/transformers-hep/JetClass/JetClass_pt_part/'+jet+'_'+ttype+'.h5  --class_label 1 --nBins 40 30 30 --nJets 10000000 --tag 1Mfromeach_403030'
        os.system(command)
    
