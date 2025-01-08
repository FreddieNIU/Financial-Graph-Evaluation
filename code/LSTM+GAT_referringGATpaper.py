
import torch
import math
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.conv import GATConv
from torch.nn import Linear, Softmax, ReLU, Sigmoid, ELU
import torch.nn as nn
from torch_geometric.utils import add_self_loops,degree
from torch_geometric.datasets import Planetoid
import ssl
import os, time
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
        # print(next(self.lstm.parameters()).is_cuda)

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
        self.gat1=GATConv(lstm_hidden_size, 256,4, edge_dim=1).double()
        # self.gat1 = initGATConv(self.gat1)
        self.gat2=GATConv(256*4,256,4, edge_dim=1).double()
        # self.gat2 = initGATConv(self.gat2)
        self.gat3=GATConv(256*4,3,16, concat=False, edge_dim=1).double()
        # self.gat3 = initGATConv(self.gat3)
        self.elu=ELU().double()
        # self.output = Linear(64, 1)
        self.sigmoid = Sigmoid()
        self.softmax = Softmax()

    def forward(self,data):
        x,edge_index, edge_attr = data.x.float(), data.edge_index.long(), data.edge_attr.double()
        x=self.lstm(x).double().to(device)
        x=self.gat1(x=x,edge_index=edge_index, edge_attr=edge_attr)
        x=self.elu(x)
        x=self.gat2(x=x,edge_index=edge_index, edge_attr=edge_attr)
        x=self.elu(x)
        x=self.gat3(x=x,edge_index=edge_index, edge_attr=edge_attr).float()
        # x=self.sigmoid(x)
        x=self.softmax(x)
        return x

if __name__=="__main__":

    # ERGATFOLDERPATH = '/home/yingjie/Year_1/ERGAT/'
    ERGATFOLDERPATH = '/media/yingjie_niu/data/Year_2/ERGAT-master/'
    args = parser.parse_args()
    # os.environ['CUDA_VISIBLE_DEVICES'] = '3'
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    ssl._create_default_https_context = ssl._create_unverified_context
    seed = 1.3423
    torch.manual_seed(seed)
    if args.dynamic == 0:
        # dynamic graph
        # root = ERGATFOLDERPATH+'data/202209_202310/dynamic/Node3_Edge1_Normmean_Threshold0.1_WeightedEdgeCorrelation_thgnn_True_2'
        root = ERGATFOLDERPATH+'data/202209_202310/dynamic/Node3_Edge1_Normmin-max_Threshold1.0_WeightedEdgeCounter_text_True_2'
    else:
        # static graph
        root = ERGATFOLDERPATH+'data/202209_202310/static/Node3_Edge1_Normmin-max_Threshold0.0_WeightedEdgeCounter_text_True'
    # %%    
    writer = SummaryWriter(log_dir=ERGATFOLDERPATH+'code/runs/'+root.split('/')[6]+'/'+root.split('/')[7]+'/'+root.split('/')[8]+'/')
    dataset = MyDataset(root=root)
    earlystopper = EarlyStopper(patience=30)
    timestamp = time.strftime("-%Y%m%d-%H%M%S", time.localtime())  #time stamp
    logger = Logger(filename=timestamp)
    # %%
    dev_len, test_len = int(len(dataset)/10), int(len(dataset)/10)
    train_len = len(dataset)-dev_len-test_len

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.load_path == 'No':
        model = Net(lstm_input_size=dataset.num_node_features, lstm_hidden_size=64, device=device).to(device)
    else:
        model = torch.load(args.load_path)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    criterion = nn.CrossEntropyLoss(weight=torch.tensor([6.11, 1.46, 6.43]))
    # criterion = MultiClassFocalLossWithAlpha(device=device)
    criterion.to(device)
    logger.write(f"Dataset: {root} \n")
    for arg in vars(args):
        logger.write(f"{arg} = {getattr(args, arg)} \n")
    for epoch in tqdm(range(args.epoch)):
        logger.write(f'Epoch: {epoch} \n')
        # if epoch == 100:
        #     optimizer = torch.optim.Adam(model.parameters(), lr=args.lr/10, weight_decay=5e-4)
        # if epoch == 400:
        #     optimizer = torch.optim.Adam(model.parameters(), lr=args.lr/100, weight_decay=5e-4)
        
        # train
        train_loss = 0
        model.train()
        for i in range(train_len):
            # 每一张图先整体输入进模型并更新一次参数           
            data = dataset[i]
            data = get_connected_subgraph(data)
            target = YToThreeClasses(data).to(device) #(num_node, 1)
            data.to(device)
            # target = torch.concat([1* (data.y > 0)], dim=1).reshape(-1).float().to(device)  # (num_node, )
            out = model(data)
            # out = out.squeeze(dim=-1) # (num_node, )
            # print(out[0:5].argmax(dim=1), target[0:5])     
            loss = criterion(out, target)
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()
            train_loss += loss.item()
            # # 每一张图再被分成tensor_size个节点为一组的子图， 依次输入进模型并更新参数，增加模型对同一张图内不同节点的辨识能力
            # data = dataset[i]
            # data = get_connected_subgraph(data)
            # subgraph_index_list = generate_tensor_list(num_nodes=data.x.shape[0], tensor_size=30)
            # for index in subgraph_index_list:
            #     subgraph = data.subgraph(index)
            #     target = YToThreeClasses(subgraph).to(device) #(num_node, 1)
            #     subgraph.to(device)
            #     # target = torch.concat([1* (data.y > 0), 1* (data.y < 0)], dim=1).float().to(device) # (num_node, 2)
            #     # target = torch.concat([1* (subgraph.y > 0)], dim=1).reshape(-1).float().to(device)  # (num_node, )
            #     out = model(subgraph)
            #     # out = out.squeeze(dim=-1) # (num_node, )
            #     print(out[0:5].argmax(dim=1), target[0:5])     
            #     loss = criterion(out, target)
            #     optimizer.zero_grad()
            #     loss.backward(retain_graph=True)
            #     optimizer.step()
            #     train_loss += loss.item()
        logger.write(f'Train Loss: {train_loss/train_len} \n')

        # dev
        model.eval()
        dev_loss = 0
        for i in range(dev_len):
            data = dataset[train_len+i]
            data = get_connected_subgraph(data)
            target = YToThreeClasses(data).to(device) #(num_node, 1)           
            data.to(device)
            # target = torch.concat([1* (data.y > 0), 1* (data.y < 0)], dim=1).float().to(device)
            # target = torch.concat([1* (data.y > 0)], dim=1).reshape(-1).float().to(device)  # (num_node, )
            out = model(data)
            # out = out.squeeze(dim=-1) # (num_node, )
            loss = criterion(out, target)
            dev_loss += loss.item()
        logger.write(f'Dev Loss: {dev_loss/dev_len} \n')
        if earlystopper.early_stop(validation_loss=dev_loss/dev_len):
            logger.write(f'Early stop! Min loss: {earlystopper.min_validation_loss} \n')
            logger.write(f'Early stop model validation loss: {dev_loss/dev_len} \n')
            break
        writer.add_scalar(tag='Loss/train', scalar_value=train_loss, global_step=epoch)
        writer.add_scalar(tag='Loss/dev', scalar_value=dev_loss, global_step=epoch)

    # 二分类
    # with torch.no_grad():
    #     # roc_auc = ROC_AUC()
    #     TP, TN, FP, FN = 0,0,0,0
    #     AUC = 0
    #     for i in range(test_len):
    #         data = dataset[train_len+dev_len+i]
    #         data = get_connected_subgraph(data)
    #         data.to(device)
    #         # target = torch.concat([1* (data.y >= 0), 1* (data.y < 0)], dim=1).float().to(device)
    #         # target = torch.concat([1* (data.y > 0)], dim=1).reshape(-1).float().to(device)  # (num_node, )
    #         out = model(data)
    #         # out = out.squeeze(dim=-1) # (num_node, )
    #         # out=softmax_to_label(out, device)
    #         out=out.to(device)
    #         logger.write(f"{out[0:5]} {target[0:5]} \n")
    #         tp, tn, fp, fn  = calculate_confusion_matrix(label=target, pred=out)
    #         TP += tp
    #         TN += tn
    #         FP += fp
    #         FN += fn
    #         AUC += roc_auc_score(target.cpu(), out.cpu())
    #     logger.write(f"TP:{TP}, FP:{FP}, TN:{TN}, FN:{FN}")
    #     precision = TP/(TP+FP) if TP+FP!=0 else 0
    #     recall = TP/(TP+FN) if TP+FN!=0 else 0
    #     TNR = TN/(TN+FP) if TN+FP!= 0 else 0
    #     f1 = 2 * precision * recall / (precision+recall) if precision+recall!=0 else 0
    #     # Matthews Correlation Coefficient (MCC) to avoid bias due to data skew
    #     MCC = (TP*TN-FP*FN) / (math.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))) if (TP+FP)*(TP+FN)*(TN+FP)*(TN+FN)!=0 else 0
    #     acc = (TP+TN)/(TP+TN+FP+FN)
    #     AUC = AUC/test_len
    #     logger.write(f'Accuracy: {acc}, Precision: {precision}, Recall: {recall}, TNR: {TNR}, F1: {f1}, MCC: {MCC}, AUC: {AUC} \n')

    # 三分类
    with torch.no_grad():
        y_pred, y_true = torch.tensor([]), torch.tensor([])
        pred_probabilities = torch.tensor([])
        auc_y_true = torch.tensor([])
        for i in range(train_len):
            data = dataset[i]
            data = get_connected_subgraph(data)
            target = YToThreeClasses(data, argmax=False) #(num_node, 3)
            auc_y_true = torch.concat([auc_y_true, target])
            target = target.argmax(dim=1)
            data.to(device)
            out = model(data)
            out=out.to('cpu')
            pred_probabilities = torch.concat([pred_probabilities,out])
            logger.write(f"{out[0:5]} {target[0:5]} \n")
            out = out.argmax(dim=1)
            logger.write(f"{out[0:5]} {target[0:5]} \n")
            y_true = torch.concat([y_true, target])
            y_pred = torch.concat([y_pred, out])
            # logger.write(f"{y_true} {y_pred} \n")

        accuracy, macro_precision, macro_recall, f1, MCC, AUC = EvaluateThreeClass(y_true, y_pred, auc_y_true, pred_probabilities)
        logger.write(f'Train Accuracy: {accuracy}, Macro Precision: {macro_precision}, Macro Recall: {macro_recall}, Macro F1: {f1}, MCC: {MCC}, AUC: {AUC} \n')

        y_pred, y_true = torch.tensor([]), torch.tensor([])
        pred_probabilities = torch.tensor([])
        auc_y_true = torch.tensor([])
        for i in range(test_len):
            data = dataset[train_len+dev_len+i]
            data = get_connected_subgraph(data)
            target = YToThreeClasses(data, argmax=False) #(num_node, 3)
            auc_y_true = torch.concat([auc_y_true, target])
            target = target.argmax(dim=1)
            data.to(device)
            out = model(data)
            out=out.to('cpu')
            pred_probabilities = torch.concat([pred_probabilities,out])
            logger.write(f"{out[0:5]} {target[0:5]} \n")
            out = out.argmax(dim=1)
            logger.write(f"{out[0:5]} {target[0:5]} \n")
            y_true = torch.concat([y_true, target])
            y_pred = torch.concat([y_pred, out])
            # logger.write(f"{y_true} {y_pred} \n")

        accuracy, macro_precision, macro_recall, f1, MCC, AUC = EvaluateThreeClass(y_true, y_pred, auc_y_true, pred_probabilities)
        logger.write(f'Test Accuracy: {accuracy}, Macro Precision: {macro_precision}, Macro Recall: {macro_recall}, Macro F1: {f1}, MCC: {MCC}, AUC: {AUC} \n')


    if args.save:
        torch.save(model, root+'/mcc'+str(MCC)+'_f1'+str(f1)+'_auc'+str(AUC)+'_epoch'+str(args.epoch)+'_lr'+str(args.lr)+'_GAT.pt')



# %%
