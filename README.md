
# The fundamental limit of jet tagging


Repo corresponding to ref. arXiv:2411.02628
The code have been slighlty adapted from https://github.com/tfinke95/transformer-hep


 # Installation

```bash
conda env create --name new_env_name --file torch_env_pip_env.yml
conda activate new_env_name

```

 # Data preprocessing
 
To creat preprocessed training data, using the same binning as in the paper. You will need to copy TrueJetClass/preprocessing_bins/ to . .
 
 
```python
python preprocess_jetclass_onebinner.py --input_file  <jet-class-data>   --nBins 40 30 30 --nJets 10000000

```
# Train generative transformer

```python

python train.py --data_path TrueJetClass/OneBin/TTBar_train___1Mfromeach_403030.h5 --model_path  <model-dir> --log_dir <model-dir>  --output linear --num_const 128 --num_epochs 5  --lr 0.001 --lr_decay 1e-06 --batch_size 100 --num_events 1000 --dropout 0 --num_heads 4 --num_layers 8 --num_bins 41 31 31 --weight_decay 1e-05 --hidden_dim 256 --end_token --start_token  --name_sufix YONFFAQ --num_events_val 5000 --checkpoint_steps 1200000

```

# Sample jets

```python
python sample_jets.py --model_dir <model-dir> --savetag <samples-tag-name>  --num_samples 100 --num_const 128 --trunc 5000 --batchsize 100 --model_name best
```

# Density estimation

```python

python evaluate_probabilities.py --model GenModels/TTBar_models/TTBar_run_test__part_pt_1Mfromeach_403030_test_2_343QU3V  --data top/discretized/test_nsamples200000_trunc_5000_nonan_top_1Mfromeach_403030.h5 --tag <evals-tag-name> --num_const 128  --num_events 128 --fixed_samples

```

# Train Baseline-Transformer classifier


```python
python train_classifier.py   --log_dir <c-model-dir> --bg qcd/discretized/samples__nsamples1000000_trunc_5000.h5 --sig top/discretized/train_nsamples1000000_trunc_5000_nonan_top_1Mfromeach_403030.h5 --num_const 128 --num_epochs 5  --lr 0.001 --batch_size 100 --num_events 100 --dropout 0.0 --num_heads 4 --num_layers 8 --hidden_dim 256 --name_sufix 4PTCYEG --fixed_samples

```


# Test classifier

```python
python test_classifier.py --data_path_1 top/discretized/train_nsamples1000000_trunc_5000_nonan_top_1Mfromeach_403030.h5 --data_path_2 qcd/discretized/samples__nsamples200000_trunc_5000.h5 --model_dir <c-model-dir>  --num_events 100 --num_const 128
```
