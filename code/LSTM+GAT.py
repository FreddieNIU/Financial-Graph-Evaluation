
import torch
import math
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.conv import GATConv
from torch.nn import Linear, Softmax, ReLU, Sigmoid
import torch.nn as nn
from torch_geometric.utils import add_self_loops,degree
from torch_geometric.datasets import Planetoid
import ssl
import os
import torch.nn.functional as F
from tqdm import tqdm
from Generate_dataset import MyDataset
from torch.utils.tensorboard import SummaryWriter
import argparse
from sklearn.metrics import roc_auc_score
from utils import *

parser = argparse.ArgumentParser()

parser.add_argument('--epoch', type=int, default='300',
                    help='Number of epochs')
parser.add_argument('--lr', type=float, default='0.0001',
                    help='Learning Rate')
parser.add_argument('--dynamic', type=int, default='0',
                    help='0: dynamic graph, 1: static graph')
parser.add_argument('--save',action='store_true',
                    help='default is False, set True if want to save the model')
parser.add_argument('--load_path',type=str, default='No',
                    help='path of the model file to be loaded')

def softmax_to_label(out, device):
    out = out.to('cpu')
    index = torch.argmax(out, dim=1)
    index = index.reshape(len(index),-1)
    value = torch.ones_like(out)
    # print(index)
    predicted = torch.zeros_like(out)
    predicted = predicted.scatter_(1,torch.LongTensor(index),value)
    return predicted

class FocalLoss(nn.Module):
    
    def __init__(self,gamma = 2.5, alpha = 0.1):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        
    def forward(self,y_pred,y_true):
        bce = nn.BCELoss(reduction= "none")(y_pred,y_true)
        p_t = (y_true * y_pred) + ((1 - y_true) * (1 - y_pred))
        alpha_factor = y_true * self.alpha + (1 - y_true) * (1 - self.alpha)
        modulating_factor = torch.pow(1.0 - p_t,self.gamma)
        loss = torch.mean(alpha_factor * modulating_factor * bce)
        return loss

class MyLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, device):
        super(MyLSTM, self).__init__()
        self.device=device
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True).to(device)
        # self.fc = nn.Linear(hidden_size, output_size)
        print(next(self.lstm.parameters()).is_cuda)

    def forward(self, x):
        h0 = torch.zeros(1, x.size(0), self.hidden_size).to(self.device)
        c0 = torch.zeros(1, x.size(0), self.hidden_size).to(self.device)
        h0 = h0.float()
        c0 = c0.float()
        out, (ht, _) = self.lstm(x, (h0, c0))
        # out = self.fc(out[:, -1, :])
        return ht[0]

class Net(torch.nn.Module):
    def __init__(self, lstm_input_size, lstm_hidden_size, device):
        super(Net,self).__init__()
        self.lstm = MyLSTM(lstm_input_size, lstm_hidden_size, device)
        self.gat1=GATConv(lstm_hidden_size, 32,2, edge_dim=1).double()
        self.gat2=GATConv(64,32,2, edge_dim=1).double()
        # self.gat3=GATConv(64,32,2, edge_dim=1).double()

        self.activate=ReLU().double()
        self.output = Linear(64, 1)
        # self.output2 = Linear(8, 1)
        # self.softmax = Softmax(dim=1)
        self.sigmoid = Sigmoid()

    def forward(self,data):
        x,edge_index, edge_attr = data.x.float(), data.edge_index.long(), data.edge_attr.double()
        x=self.lstm(x).double().to(device)
        x=self.gat1(x=x,edge_index=edge_index, edge_attr=edge_attr)
        x=self.activate(x)
        # x=F.dropout(x, p=0.5, training=self.training)
        x=self.gat2(x=x,edge_index=edge_index, edge_attr=edge_attr).float()
        x=self.activate(x)
        # x=self.gat3(x=x,edge_index=edge_index, edge_attr=edge_attr).float()
        # x=self.activate(x)
        x=self.output(x)
        # x=self.output2(x)
        x=self.sigmoid(x)
        return x

if __name__=="__main__":

    args = parser.parse_args()
    
    # os.environ['CUDA_VISIBLE_DEVICES'] = '3'
    os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
    ssl._create_default_https_context = ssl._create_unverified_context
    seed = 1.3423
    torch.manual_seed(seed)
    if args.dynamic == 0:
        # dynamic graph
        # root = '/home/yingjie/Year_1/ERGAT/data/Graph_Dataset2'
        # root = '/home/yingjie/Year_1/ERGAT/data/202202_202208/Graph_Dataset_meanNorm_threshold0.1_CorrelationWeightedEdge_dynamic_forRNN'
        # root = '/home/yingjie_niu/Year_2/Entity_GCN/data/202209_202303/Graph_Dataset_meanNorm_threshold3_NoWeightedEdge_dynamic'
        root = '/media/yingjie_niu/data/Year_2/ERGAT-master/data/202209_202310/dynamic/Node3_Edge1_Normmin-max_Threshold0.0_WeightedEdgeCounter_text_True_2'
    else:
        # static graph
        # root = '/home/yingjie/Year_1/ERGAT/data/Graph_Dataset'
        # root = '/home/yingjie/Year_1/ERGAT/data/Graph_Dataset_meanNorm_threshold3_NoWeightedEdge_static'
        root = '/media/yingjie_niu/data/Year_2/ERGAT-master/data/202209_202310/static/Node3_Edge1_Normmin-max_Threshold0.0_WeightedEdgeCounter_text_True'
    # %%    
    writer = SummaryWriter(log_dir='/media/yingjie_niu/data/Year_2/ERGAT-master/code/runs/'+root.split('/')[6]+'/'+root.split('/')[7]+'/'+root.split('/')[8]+'/')
    dataset = MyDataset(root=root)
    # %%
    dev_len, test_len = int(len(dataset)/10), int(len(dataset)/10)
    train_len = len(dataset)-dev_len-test_len

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.load_path == 'No':
        model = Net(lstm_input_size=dataset.num_node_features, lstm_hidden_size=16, device=device).to(device)
    else:
        model = torch.load(args.load_path)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)

    criterion = nn.BCELoss()

    
    for epoch in tqdm(range(args.epoch)):

        if epoch == 100:
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr/10, weight_decay=5e-4)
        if epoch == 400:
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr/100, weight_decay=5e-4)
        
        
        # train
        train_loss = 0
        model.train()
        for i in range(train_len):
            # print(model.gat1.att_src)
            data = dataset[i]
            # print(data.y)
            data.to(device)
            # target = torch.concat([1* (data.y > 0), 1* (data.y < 0)], dim=1).float().to(device)
            target = torch.concat([1* (data.y > 0)], dim=1).reshape(-1).float().to(device)  # (num_node, )
            out = model(data)
            out = out.squeeze(dim=-1) # (num_node, )
            # print(out[0:5], target[0:5])     
            loss = criterion(out, target)
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()
            train_loss += loss.item()
        # print('Train Loss: ', train_loss/train_len)

        # dev
        model.eval()
        dev_loss = 0
        for i in range(dev_len):
            data = dataset[train_len+i]
            data.to(device)
            # target = torch.concat([1* (data.y > 0), 1* (data.y < 0)], dim=1).float().to(device)
            target = torch.concat([1* (data.y > 0)], dim=1).reshape(-1).float().to(device)  # (num_node, )
            out = model(data)
            out = out.squeeze(dim=-1) # (num_node, )
            loss = criterion(out, target)
            dev_loss += loss.item()
        # print('Dev Loss: ', dev_loss/dev_len)
        writer.add_scalar(tag='Loss/train', scalar_value=train_loss, global_step=epoch)
        writer.add_scalar(tag='Loss/dev', scalar_value=dev_loss, global_step=epoch)

    # model.eval()
    with torch.no_grad():
        # roc_auc = ROC_AUC()
        TP, TN, FP, FN = 0,0,0,0
        AUC = 0
        for i in range(test_len):
            data = dataset[train_len+dev_len+i]
            data.to(device)
            # target = torch.concat([1* (data.y >= 0), 1* (data.y < 0)], dim=1).float().to(device)
            target = torch.concat([1* (data.y > 0)], dim=1).reshape(-1).float().to(device)  # (num_node, )
            out = model(data)
            out = out.squeeze(dim=-1) # (num_node, )
            # out=softmax_to_label(out, device)
            out=out.to(device)
            print(out[0:5], target[0:5])
            # pos, neg = torch.tensor([1.,0.]).to(device), torch.tensor([0.,1.]).to(device)
            # for i in range(len(out)):
            #     if (out[i][0]==pos[0] and target[i][0]==pos[0]) and (out[i][1]==pos[1] and target[i][1]==pos[1]):
            #         TP+=1
            #     elif (out[i][0]==neg[0] and target[i][0]==neg[0]) and (out[i][1]==neg[1] and target[i][1]==neg[1]):
            #         TN+=1
            #     elif (out[i][0]==pos[0] and target[i][0]==neg[0]) and (out[i][1]==pos[1] and target[i][1]==neg[1]):
            #         FP+=1
            #     elif (out[i][0]==neg[0] and target[i][0]==pos[0]) and (out[i][1]==neg[1] and target[i][1]==pos[1]):
            #         FN+=1
            TP, TN, FP, FN  = calculate_confusion_matrix(label=target, pred=out)
            AUC += roc_auc_score(target.cpu(), out.cpu())
        # print(TP, FP, TN, FN)
        precision = TP/(TP+FP) if TP+FP!=0 else 0
        recall = TP/(TP+FN) if TP+FN!=0 else 0
        TNR = TN/(TN+FP) if TN+FP!= 0 else 0
        f1 = 2 * precision * recall / (precision+recall) if precision+recall!=0 else 0
        # Matthews Correlation Coefficient (MCC) to avoid bias due to data skew
        MCC = (TP*TN-FP*FN) / (math.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))) if (TP+FP)*(TP+FN)*(TN+FP)*(TN+FN)!=0 else 0
        acc = (TP+TN)/(TP+TN+FP+FN)
        AUC = AUC/test_len
        print('Accuracy: ', acc, 'Precision: ', precision, 'Recall: ', recall,'TNR: ',TNR, 'F1: ', f1, "MCC: ", MCC, "AUC: ",AUC)

    if args.save:
        torch.save(model, root+'/mcc'+str(MCC)+'_f1'+str(f1)+'_auc'+str(AUC)+'_epoch'+str(args.epoch)+'_lr'+str(args.lr)+'_GAT.pt')



# %%
