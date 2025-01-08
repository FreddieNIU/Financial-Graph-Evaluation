from torch import nn
import math
import random
import torch
import numpy as np
from sklearn.metrics import roc_auc_score
import os



def reset_parameters(named_parameters):
    for i in named_parameters():
        if len(i[1].size()) == 1:
            std = 1.0 / math.sqrt(i[1].size(0))
            nn.init.uniform_(i[1], -std, std)
        else:
            nn.init.xavier_normal_(i[1])

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def metrics(trues, preds):
    trues = np.concatenate(trues,-1)
    preds = np.concatenate(preds,0)
    acc = sum(preds.argmax(-1) == trues) / len(trues)
    auc = roc_auc_score(trues,preds[:,1])
    preds = preds.argmax(-1)
    TP, TN, FP, FN = 0,0,0,0
    for i in range(len(trues)):
        if (trues[i]==preds[i] and trues[i]==1):
            TP+=1
        elif (trues[i]==preds[i] and trues[i]==0):
            TN+=1
        elif (trues[i]!=preds[i] and trues[i]==1):
            FP+=1
        elif (trues[i]!=preds[i] and trues[i]==0):
            FN+=1
    precision = TP/(TP+FP)
    recall = TP/(TP+FN)
    TNR = TN/(TN+FP)
    f1 = 2 * precision * recall / (precision+recall)
    # Matthews Correlation Coefficient (MCC) to avoid bias due to data skew
    MCC = (TP*TN-FP*FN) / (math.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN)))
    acc2 = (TP+TN)/(TP+TN+FP+FN)
    print('Accuracy: ', acc, acc2, 'Precision: ', precision, 'Recall: ', recall,'TNR: ',TNR, 'F1: ', f1, "MCC: ", MCC)
    return acc, auc

def createPath(path):
    if os.path.exists(path):
        pass
    else:
        os.makedirs(path)

