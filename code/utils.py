import torch
import math
from sklearn.metrics import roc_auc_score,accuracy_score, average_precision_score,precision_score,f1_score,recall_score,matthews_corrcoef
import numpy as np
import pandas as pd
import sys
import scipy
import matplotlib.pyplot as plt
import networkx as nx
from os import listdir
import torch_geometric as PyG
import json

# ERGATFOLDERPATH = '/home/yingjie/Year_1/ERGAT/'
ERGATFOLDERPATH = '/media/yingjie_niu/data/Year_2/ERGAT-master/'

def softmax_to_label(out, device):
    out = out.to('cpu')
    index = torch.argmax(out, dim=1)
    index = index.reshape(len(index),-1)
    value = torch.ones_like(out)
    # print(index)
    predicted = torch.zeros_like(out)
    predicted = predicted.scatter_(1,torch.LongTensor(index),value)
    return predicted

def evaluate(model, testset, test_len, device):
    CORRECT, ALL = 0, 0
    TP, TN, FP, FN = 0,0,0,0
    AUC = 0
    for i in range(test_len):
        data = testset
        data.to(device)
        target = torch.concat([1* (data.y >= 0), 1* (data.y < 0)], dim=1).float().to(device)
        out = model(data)
        out=softmax_to_label(out, device)
        out=out.to(device)
        # print(out[0:5], target[0:5])
        correct = 1*(out==target).sum()/2
        correct = correct.item()
        all = target.shape[0]
        pos, neg = torch.tensor([1.,0.]).to(device), torch.tensor([0.,1.]).to(device)
        for i in range(len(out)):
            if (out[i][0]==pos[0] and target[i][0]==pos[0]) and (out[i][1]==pos[1] and target[i][1]==pos[1]):
                TP+=1
            elif (out[i][0]==neg[0] and target[i][0]==neg[0]) and (out[i][1]==neg[1] and target[i][1]==neg[1]):
                TN+=1
            elif (out[i][0]==pos[0] and target[i][0]==neg[0]) and (out[i][1]==pos[1] and target[i][1]==neg[1]):
                FP+=1
            elif (out[i][0]==neg[0] and target[i][0]==pos[0]) and (out[i][1]==neg[1] and target[i][1]==pos[1]):
                FN+=1
        CORRECT += correct
        ALL += all
        AUC += roc_auc_score(target.cpu(), out.cpu())
    # print(TP, FP, TN, FN)
    precision = TP/(TP+FP)
    recall = TP/(TP+FN)
    TNR = TN/(TN+FP)
    f1 = 2 * precision * recall / (precision+recall)
    # Matthews Correlation Coefficient (MCC) to avoid bias due to data skew
    MCC = (TP*TN-FP*FN) / (math.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN)))
    acc = (TP+TN)/(TP+TN+FP+FN)
    AUC = AUC/test_len

    return acc, precision, recall, TNR, f1, MCC, AUC

def calculate_confusion_matrix(label, pred):
    """
    Calculates TP, TN, FP, FN based on label and target tensors.
    
    Args:
        label (torch.Tensor): Ground truth labels (binary, 0 or 1).
        pred (torch.Tensor): Predicted labels (binary, 0 or 1).
    
    Returns:
        int: True Positives (TP)
        int: True Negatives (TN)
        int: False Positives (FP)
        int: False Negatives (FN)
    """
    assert label.shape == pred.shape, "Input tensors must have the same shape"
    
    # Calculate confusion matrix elements
    TP = torch.sum((label == 1) & (pred >= 0.5))
    TN = torch.sum((label == 0) & (pred < 0.5))
    FP = torch.sum((label == 0) & (pred >= 0.5))
    FN = torch.sum((label == 1) & (pred < 0.5))
    
    return TP.item(), TN.item(), FP.item(), FN.item()

class AutoUpdateQueue():
    
    def __init__(self, maxsize=3):
        self.maxsize = maxsize
        self.list = []
    
    def __str__(self) -> str:
        return self.list.__str__()
    
    def check_maxsize(self):
        if len(self.list) >= self.maxsize:
            return True
        else:
            return False
    
    def check_empty(self):
        if len(self.list) == 0:
            return True
        else:
            return False
    
    def put(self, a):
        if self.check_maxsize():
            first_element = self.pop()
            self.list.append(a)
            # print('Auto Updated!')
        else:
            self.list.append(a)

    def get(self):
        if self.check_empty():
            return None
        else:
            first_element = self.list[0]
            return first_element
    def get_last(self):
        if self.check_empty():
            return None
        else:
            last_element = self.list[-1]
            return last_element

    def pop(self):
        if self.check_empty():
            return None
        else:
            first_element = self.list[0]
            self.list = self.list[1:]
            return first_element
        
def merge_historical_graph(historical_graph, graph_of_today, forTHGNN=False):
    """
    merge graph_of_today to historical_graph
    this function can also be used to generate daily graph if --build_edge=thgnn
    """
    AttenuateRate = 1
    merged_graph = {}
    ht_edge_idx = historical_graph['Edge_index_Transpose']
    ht_edge_attr = historical_graph['Edge_attr']
    today_edge_idx = graph_of_today['Edge_index_Transpose']
    today_edge_attr = graph_of_today['Edge_attr']
    ht_common_edge_idx = (ht_edge_idx[:,None] == today_edge_idx).all(-1).any(-1)
    today_common_edge_idx = (today_edge_idx[:,None] == ht_edge_idx).all(-1).any(-1)
    merged_edge_idx = np.concatenate([ht_edge_idx[ht_common_edge_idx],ht_edge_idx[~ht_common_edge_idx],today_edge_idx[~today_common_edge_idx]])
    # print("ht_edge_idx",ht_edge_idx.shape,"today_edge_idx",today_edge_idx.shape)
    # print("ht_edge_attr",ht_edge_attr.shape,'today_edge_attr',today_edge_attr.shape,'ht_common_edge_idx',ht_common_edge_idx.shape,'today_common_edge_idx',today_common_edge_idx.shape)
    # a = ht_edge_attr[ht_common_edge_idx]*AttenuateRate+today_edge_attr[today_common_edge_idx]
    if forTHGNN:
        merged_edge_attr = np.concatenate([ht_edge_attr[ht_common_edge_idx], ht_edge_attr[~ht_common_edge_idx], today_edge_attr[~today_common_edge_idx]])  
    else:
        merged_edge_attr = np.concatenate([ht_edge_attr[ht_common_edge_idx]*AttenuateRate+today_edge_attr[today_common_edge_idx], ht_edge_attr[~ht_common_edge_idx], today_edge_attr[~today_common_edge_idx]])
    merged_graph['Edge_index_Transpose'] = merged_edge_idx
    merged_graph['Edge_attr'] = merged_edge_attr
    return merged_graph

class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.min_delta = 0
            # self.min_delta = 0.01 * self.min_validation_loss  # 允许1% loss 浮动
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False
    

class Logger(object):
    def __init__(self, filename='default.log'):
        self.log = open(ERGATFOLDERPATH+"outputs/ModelTrainingLogs/"+filename+".log", 'a')

    def write(self, message):
        self.log.write(message)

    def flush(self):
        pass

def generate_tensor_list(num_nodes, tensor_size=10):

    """
    Generate the index tensor of subgraphs
    largest_element: number of nodes 
    tensor_size = 10, means 10 nodes in a subgraph
    """

    # Largest index is num_nodes-1
    largest_element = num_nodes-1
    # Calculate the number of tensors required
    num_tensors = (largest_element) // tensor_size + 1
    # Create an empty list to store the tensors
    batch = []
    # Loop through the number of tensors
    for i in range(num_tensors):
        start = i * tensor_size
        end = min(start + tensor_size, largest_element+1)
        tensor = torch.arange(start, end)
        batch.append(tensor)
    # Return the batch list
    return batch

def get_connected_subgraph(data):
    """
    Input is a pyg.Data
    Output is a subgraph containing only the nodes whose degree > 1
    """
    connected_nodes = torch.tensor(np.array(list(set(np.array(data.edge_index[0])))))
    connected_subgraph = data.subgraph(connected_nodes)
    return connected_subgraph

def YToThreeClasses(data, argmax=True):
    """
    Input:
    data : dataset[i] the daily graph
    Output: 

    """
    y = data.y
    mu, std = scipy.stats.norm.fit(y.numpy())
    # print("mu: ",mu, "std: ",std)
    pos = y > std
    neg = y < -std
    neutral = (y < std) & (y >- std)
    # print("Pos: ",torch.sum(pos).item(), "Neutral: ",torch.sum(neutral).item(), "Neg: ",torch.sum(neg).item())
    target = torch.concat([1* pos, 1* neutral, 1*neg], dim=1).float()
    if argmax==True:
        return target.argmax(dim=1)
    else:
        return target

def EvaluateThreeClass(y_true, y_pred, auc_y_true, pred_probabilities):
    macro_precision = precision_score(y_true, y_pred, average='macro')
    macro_recall = recall_score(y_true, y_pred, average='macro')
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    accuracy = accuracy_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)
    auc = roc_auc_score(auc_y_true, y_score=pred_probabilities)
    return accuracy, macro_precision, macro_recall, macro_f1, mcc, auc

class MultiClassFocalLossWithAlpha(torch.nn.Module):
    def __init__(self, alpha=[0.436, 0.104, 0.459], gamma=2, reduction='mean', device='cpu'):
        """
        :param alpha: 权重系数列表，三分类中第0类权重0.2，第1类权重0.3，第2类权重0.5
        :param gamma: 困难样本挖掘的gamma
        :param reduction:
        """
        super(MultiClassFocalLossWithAlpha, self).__init__()
        self.alpha = torch.tensor(alpha).to(device)
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, pred, target):
        alpha = self.alpha[target]  # 为当前batch内的样本，逐个分配类别权重，shape=(bs), 一维向量
        log_softmax = torch.log(pred) # 对模型裸输出做softmax再取log, shape=(bs, 3)
        logpt = torch.gather(log_softmax, dim=1, index=target.view(-1, 1))  # 取出每个样本在类别标签位置的log_softmax值, shape=(bs, 1)
        logpt = logpt.view(-1)  # 降维，shape=(bs)
        ce_loss = -logpt  # 对log_softmax再取负，就是交叉熵了
        pt = torch.exp(logpt)  #对log_softmax取exp，把log消了，就是每个样本在类别标签位置的softmax值了，shape=(bs)
        focal_loss = alpha * (1 - pt) ** self.gamma * ce_loss  # 根据公式计算focal loss，得到每个样本的loss值，shape=(bs)
        if self.reduction == "mean":
            return torch.mean(focal_loss)
        if self.reduction == "sum":
            return torch.sum(focal_loss)
        return focal_loss
    
def initGATConv(gat, glolot_initialization="normal"):
    """
    Initialize the GATConv layer using Glorot Initialization method
    """

    if glolot_initialization=='normal':
        torch.nn.init.xavier_normal_(gat.att_src, gain=1)
        torch.nn.init.xavier_normal_(gat.att_dst, gain=1)
        torch.nn.init.xavier_normal_(gat.att_edge, gain=1)
        torch.nn.init.xavier_normal_(gat.lin_src.weight, gain=1)
        torch.nn.init.xavier_normal_(gat.lin_dst.weight, gain=1)
        torch.nn.init.xavier_normal_(gat.lin_edge.weight, gain=1)
    elif glolot_initialization=='uniform':
        torch.nn.init.xavier_uniform_(gat.att_src, gain=1)
        torch.nn.init.xavier_uniform_(gat.att_dst, gain=1)
        torch.nn.init.xavier_uniform_(gat.att_edge, gain=1)
        torch.nn.init.xavier_uniform_(gat.lin_src.weight, gain=1)
        torch.nn.init.xavier_uniform_(gat.lin_dst.weight, gain=1)
        torch.nn.init.xavier_uniform_(gat.lin_edge.weight, gain=1)
    else:
        raise KeyError(glolot_initialization)
    return gat

def VisualizeGraphDataset(dataset, day = 0, subgraph=None, full=True):
    """
    Input:
    "dataset" : graph dataset loaded using MyDataset class
    "day" : the index of day of which we want to visualize
    "full" : whethere or not print the whole graph. "True" for whole graph, "False" for removing nodes whoes degree<2
    """
    PERIOD_START = '202209'
    PERIOD_END = '202310'
    folder_path =f'{ERGATFOLDERPATH}data/SP500/dataset{PERIOD_START}-{PERIOD_END}/'
    quantitative_fileList = listdir(folder_path+'quantitative_'+PERIOD_START+'_'+PERIOD_END)
    company_list = np.load(folder_path + 'company_list.npy')
    if subgraph is None:
        G = dataset[day]
    elif isinstance(subgraph, PyG.data.data.Data):
        G = subgraph
    else:
        raise TypeError('subgraph has to be a torch_geometric.data.data.Data type')
    Node_Idx_To_Comp = []
    for company in company_list:
        if company+'.csv' in quantitative_fileList:
            Node_Idx_To_Comp.append(company)
        else:
            print(company,'.csv not exist!')
    edges = {
        "Source":[Node_Idx_To_Comp[idx] for idx in list(np.array(G.edge_index[0]))],
        'Target':[Node_Idx_To_Comp[idx] for idx in list(np.array(G.edge_index[1]))],
        'Weight':list(np.array(G.edge_attr).reshape(-1))
    }
    G_of_day = nx.from_pandas_edgelist(edges, source='Source',target='Target',edge_attr='Weight')
    if not full:
        # remove all companies with degree = 1, but those companies in our sample list with degree 1 should be left. 
        remove = [node for node,degree in G_of_day.degree() if (degree < 2 and node in company_list)]   
        G_of_day.remove_nodes_from(remove)
    temp =list(G_of_day.degree())
    temp.sort(key=lambda x:x[1], reverse=True)
    print(temp)

    plt.figure(figsize=(20,20))
    nx.draw_networkx(G_of_day)
    print("Dataset path: ",dataset.processed_dir)
    fileList = listdir(dataset.processed_dir)
    fileList.sort()
    print('Date: ',fileList[day])
    print('Graph Metadata: ', G)
    print(f"All Nodes: {G.x.shape[0]} Connected Nodes: {len(G_of_day.nodes)} (" + '%.4g'%(len(G_of_day.nodes)/G.x.shape[0]*100) + f"%) Edges: {len(G_of_day.edges)} ") 
    return G_of_day

def Describe(G_of_day):
    return pd.DataFrame(G_of_day.degree(), columns=['Node','Degree']).describe()