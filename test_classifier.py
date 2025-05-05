import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.metrics import roc_auc_score

from preprocess import preprocess_dataframe

from argparse import ArgumentParser
from tqdm import tqdm
import os
import pandas as pd

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

torch.multiprocessing.set_sharing_strategy('file_system')


def load_model(name):
    model = torch.load(os.path.join(args.model_dir, f'model_{name}.pt'))
    return model


def parse_input():
    parser = ArgumentParser()
    parser.add_argument("--model_dir", type=str, default='models/test', help="Model directory")
    parser.add_argument("--data_path_1", type=str, default='/hpcwork/bn227573/top_benchmark/', help="Path to training data file")
    parser.add_argument("--data_path_2", type=str, default='/hpcwork/bn227573/qcd_benchmark/', help="Path to training data file")

    parser.add_argument("--num_workers", type=int, default=1, help="Number of workers")
    parser.add_argument("--batch_size", type=int, default=128, help="Number of workers")

    parser.add_argument("--num_const", type=int, default=100, help="Number of constituents")
    parser.add_argument("--limit_const", action="store_true", help="Only use jets with at least num_const constituents")
    parser.add_argument("--num_events", type=int, default=10000, help="Number of events for training")
    parser.add_argument("--num_bins", type=int, nargs=3, default=[41, 31, 31], help="Number of bins per feature")
    parser.add_argument("--reverse", action='store_true', help="Whether to reverse pt order")

    parser.add_argument("--pred_name", type=str, default='predictions_test.npz', help="predicitons name")
    
    args = parser.parse_args()
    return args


def load_data(path1,path2, n_events):
    
            
    
    df = pd.read_hdf(path1, 'discretized', stop=n_events)
    
    #start_value=random.randint(0,1000000-args.num_events)
    #df = pd.read_hdf(path1, key="discretized", start=start_value, stop=start_value+args.num_events)
    
    x, padding_mask, _ = preprocess_dataframe(df, num_features=num_features,
                                num_bins=num_bins,
                                to_tensor=True,
                                num_const=args.num_const,
                                reverse=args.reverse,
                                limit_nconst=args.limit_const)
    labels = torch.zeros(len(x))

    df = pd.read_hdf(path2, 'discretized', stop=n_events)
    x1, padding_mask1, _ = preprocess_dataframe(df, num_features=num_features,
                                num_bins=num_bins,
                                to_tensor=True,
                                num_const=args.num_const,
                                reverse=args.reverse,
                                limit_nconst=args.limit_const)
                                
                                
                                
    labels = torch.concat((labels, torch.ones(len(x1))))
    x = torch.concat((x, x1), dim=0)
    padding_mask = torch.concat((padding_mask, padding_mask1), dim=0)

    train_dataset = TensorDataset(x, padding_mask, labels)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
    )
    print(x.shape)
    return train_loader
    
    
    

def plot_roc_curve(y_true, y_score,model_dir):
    # Compute ROC curve and ROC area for each class
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    # Plot ROC curve
    plt.figure()
    lw = 2
    plt.plot(tpr,1/fpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    #plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')

    plt.xlabel('True Positive Rate')
    plt.ylabel('False Positive Rate')
    plt.yscale('log')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.savefig(model_dir+'/roc_test_1fpr.png')
    return


def saveAUCscore(model_dir,auc_score):

    file=open(model_dir+'/auc.txt','w')
    file.write('auc_score')
    file.write('\n')
    file.write(str(auc_score))
    file.write('\n')
    file.close()

    return

if __name__ == '__main__':
    args = parse_input()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Running on device: {device}")

    num_features = 3
    num_bins = tuple(args.num_bins)

    print(f"Using bins: {num_bins}")
    print(f"{'Not' if not args.reverse else ''} reversing pt order")

    # load and preprocess data
    print(f"Loading test set")
    test_loader = load_data(args.data_path_1,args.data_path_2 , args.num_events)

    # construct model
    model = load_model('best')
    print("Loaded model")
    model.to(device)
    model.eval()

    loss_list = []
    prediction_list = []
    label_list = []
    logits_list=[]
    min_val_loss = np.inf
    with torch.no_grad():
        for x, padding_mask, label in tqdm(test_loader, total=len(test_loader), desc=f'Testing'):
            label_list.append(label.detach().numpy())
            x = x.to(device)
            padding_mask = padding_mask.to(device)
            label = label.to(device)

            #with torch.no_grad():
            #with torch.cuda.amp.autocast():
            logits = model(x, padding_mask)
            predictions = torch.sigmoid(logits)
            loss = model.loss(logits, label.view(-1, 1))

            loss_list.append(loss.cpu().detach().numpy())
            prediction_list.append(predictions.cpu().detach().numpy())
            logits_list.append(logits.cpu().detach().numpy())

    predictions = np.concatenate(prediction_list, axis=0)
    logits_all=np.concatenate(logits_list, axis=0)
    label_all = np.concatenate(label_list, axis=0)

    predictions=predictions[:,0]

    auc_score=roc_auc_score(label_all, predictions)
    print(auc_score)
    
    plot_roc_curve(label_all, predictions,args.model_dir)
    saveAUCscore(args.model_dir,auc_score)
    
    np.savez(os.path.join(args.model_dir, args.pred_name),
            predictions=predictions,
            labels=label_all,
            logits=logits_all)
