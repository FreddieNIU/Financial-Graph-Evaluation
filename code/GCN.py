
import torch
import math
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn import GCNConv
from torch.nn import Linear, Softmax, ReLU, Sigmoid
import torch.nn as nn
from torch_geometric.utils import add_self_loops,degree
from torch_geometric.datasets import Planetoid
import ssl
import torch.nn.functional as F
from Generate_dataset import MyDataset
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--epoch', type=int, default='10',
                    help='Number of epochs')
parser.add_argument('--lr', type=float, default='0.0001',
                    help='Learning Rate')
parser.add_argument('--dynamic', type=int, default='0',
                    help='0: dynamic graph, 1: static graph')

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
    
class Net(torch.nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.gat1=GCNConv(dataset.num_node_features,8,dropout=0.4)
        self.gat2=GCNConv(16,8,dropout=0.4)
        # self.gat3=GATConv(32,8,dropout=0.4)
        self.activate=Sigmoid()
        self.output = Linear(8, 2)
        self.softmax = Softmax(dim=1)

    def forward(self,data):
        x,edge_index=data.x.float(), data.edge_index
        x=self.gat1(x,edge_index)
        x=self.activate(x)
        # x=self.gat2(x,edge_index)
        # x=self.activate(x)
        # x=self.gat3(x,edge_index)
        # x=self.activate(x)
        x=self.output(x)
        x=self.softmax(x)
        return x

if __name__=="__main__":
    args = parser.parse_args()

    ssl._create_default_https_context = ssl._create_unverified_context
    seed = 1.3423
    torch.manual_seed(seed)
    if args.dynamic == 0:
        # dynamic graph
        root = '/home/yingjie/Year_1/ERGAT/data/Graph_Dataset2'
    else:
        # static graph
        root = '/home/yingjie/Year_1/ERGAT/data/Graph_Dataset'
    # %%    
    dataset = MyDataset(root=root)
    # %%
    dev_len, test_len = 10, 10
    train_len = len(dataset)-dev_len-test_len

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Net().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    criterion = FocalLoss()

    model.train()
    for epoch in range(args.epoch):
        print('Epoch: ', epoch)
        train_loss = 0
        optimizer.zero_grad()

        # train
        for i in range(train_len):
            print(model.gat1.lin.weight)
            data = dataset[i]
            data.to(device)
            target = torch.concat([1* (data.y > 0), 1* (data.y < 0)], dim=1).float().to(device)
            out = model(data)
            # print(out[0:5], target[0:5])
            loss = criterion(out, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        print('Train Loss: ', train_loss/train_len)

        # dev
        dev_loss = 0
        for i in range(dev_len):
            data = dataset[train_len+i]
            data.to(device)
            target = torch.concat([1* (data.y > 0), 1* (data.y < 0)], dim=1).float().to(device)
            out = model(data)
            dev_loss += loss.item()
        print('Dev Loss: ', dev_loss/dev_len)

    model.eval()

    CORRECT, ALL = 0, 0
    TP, TN, FP, FN = 0,0,0,0
    for i in range(test_len):
        data = dataset[train_len+dev_len+i]
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
    print(TP, FP, TN, FN)
    precision = TP/(TP+FP) if TP!=0 else 0
    recall = TP/(TP+FN) if TP!=0 else 0
    TNR = TN/(TN+FP) if TN!=0 else 0
    f1 = 2 * precision * recall / (precision+recall) if (precision!=0 or recall!=0) else 0
    print('Test Accuracy: ', (TP+TN)/(TP+TN+FP+FN), 'Precision: ', precision, 'Recall: ', recall,'TNR: ',TNR, 'F1: ', f1)



