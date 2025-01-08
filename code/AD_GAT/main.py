from Model import *
from utils import *
import pickle
import torch
from torch import optim
import argparse

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

parser = argparse.ArgumentParser()

parser.add_argument('--task', type=int, default='1',
                    help='0 Regression. 1 Classification')
parser.add_argument('--grid-search', type=int, default='0',
                    help='0 False. 1 True')
parser.add_argument('--soft-training', type=int, default='0',
                    help='0 False. 1 True')
parser.add_argument('--sample-discrimination', type=int, default='0',
                    help='0 False. 1 True')
parser.add_argument('--optim', type=int, default='1',
                    help='0 SGD. 1 Adam')
parser.add_argument('--eval', type=int, default='1',
                    help='if set the last day as eval')
parser.add_argument('--max-epoch', type=int, default='300',
                    help='Training max epoch')
parser.add_argument('--wait-epoch', type=int, default='30',
                    help='Training min epoch')
parser.add_argument('--eta', type=float, default='1e-4',
                    help='Early stopping')
parser.add_argument('--lr', type=float, default='5e-4',
                    help='Learning rate ')
parser.add_argument('--device', type=str, default='2',
                    help='GPU to use')
parser.add_argument('--heads-att', type=int, default='6',
                    help='attention heads')
parser.add_argument('--hidn-att', type=int, default='60',
                    help='attention hidden nodes')
parser.add_argument('--hidn-rnn', type=int, default='360',
                    help='rnn hidden nodes')
parser.add_argument('--weight-constraint', type=float, default='0',
                    help='L2 weight constraint')
parser.add_argument('--rnn-length', type=int, default='30',
                    help='rnn length')
parser.add_argument('--dropout', type=float, default='0.2',
                    help='dropout rate')
parser.add_argument('--clip', type=float, default='0.25',
                    help='rnn clip')
parser.add_argument('--infer', type=float, default='1',
                    help='if infer relation')
parser.add_argument('--relation', type=str, default='None',
                    help='all, competitor, customer, industry, stratigic, supply')
parser.add_argument('--save', type=bool, default=True,
                    help='save model')

def load_dataset(DEVICE):
    # path = '/home/yingjie_niu/Year_2/ERGAT-master/data/ADGAT_Data/202202_202208/static_threshold1_NodeFeature3/'
    path = '/home/yingjie/Year_1/ERGAT/data/ADGAT_Data/202202_202208/static_threshold1_NodeFeature3/'
    # path = '/home/yingjie/Year_1/AAAI-21-ADGAT/data/'
    with open(path+'x_numerical.pkl', 'rb') as handle:
        markets = pickle.load(handle) # 730,198,5
    with open(path+'y_.pkl', 'rb') as handle:
        y_load = pickle.load(handle) # 730, 198
    with open(path+'x_textual.pkl', 'rb') as handle:
        stock_sentiments = pickle.load(handle) # 730,198,8

    markets = markets.astype(np.float64)
    x = torch.tensor(markets, device=DEVICE)
    x.to(torch.double)
    x_sentiment = torch.tensor(stock_sentiments, device=DEVICE)
    x_sentiment.to(torch.double)
    if args.relation != "None":
        # with open(path+'relations/' + args.relation + '_relation.pkl', 'rb') as handle:
        with open(path+'relations/relationOf' + args.relation + '.pkl', 'rb') as handle:
            relation_static = pickle.load(handle)
        relation_static = torch.tensor(relation_static, device=DEVICE)
        relation_static.to(torch.double)
    else:
        relation_static = None
    y = torch.tensor(y_load, device=DEVICE)
    y = (y>0).to(torch.long)

    return x, y, x_sentiment, relation_static

def train(model, x_train, x_sentiment_train, y_train, relation_static = None):
    model.train()
    seq_len = len(x_train)
    train_seq = list(range(seq_len))[rnn_length:]
    random.shuffle(train_seq)
    total_loss = 0
    total_loss_count = 0
    batch_train = 15
    for i in train_seq:
        output = model(x_train[i - rnn_length + 1: i + 1], x_sentiment_train[i - rnn_length + 1: i + 1],  relation_static = relation_static)
        loss = criterion(output, y_train[i])
        loss.backward()
        total_loss += loss.item()
        total_loss_count += 1
        if total_loss_count % batch_train == batch_train - 1:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()
            optimizer.zero_grad()
    if total_loss_count % batch_train != batch_train - 1:
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()
    return total_loss / total_loss_count

def evaluate(model, x_eval, x_sentiment_eval, y_eval, relation_static = None):
    model.eval()
    seq_len = len(x_eval)
    seq = list(range(seq_len))[rnn_length:]
    preds = []
    trues = []
    for i in seq:
        output = model(x_eval[i - rnn_length + 1: i + 1], x_sentiment_eval[i - rnn_length + 1: i + 1], relation_static = relation_static)
        output = output.detach().cpu()
        preds.append(np.exp(output.numpy()))
        trues.append(y_eval[i].cpu().numpy())
    acc, auc = metrics(trues, preds)
    return acc,  auc


if __name__=="__main__":
    args = parser.parse_args()
    DEVICE = "cuda:" + args.device
    criterion = torch.nn.NLLLoss()
    set_seed(1017)
    if args.relation != "None":
        static = 1
        pass
    else:
        static = 0
        relation_static = None
    # load dataset
    print("loading dataset")
    x, y, x_sentiment, relation_static = load_dataset(DEVICE)
    # hyper-parameters
    NUM_STOCK = x.size(1)
    D_MARKET = x.size(2)
    D_NEWS = x_sentiment.size(2)
    MAX_EPOCH =  args.max_epoch
    infer = args.infer
    hidn_rnn = args.hidn_rnn
    heads_att = args.heads_att
    hidn_att= args.hidn_att
    lr = args.lr
    rnn_length = args.rnn_length
    t_mix = 0 
    #train-test split
    x_train = x[: -24]
    x_eval = x[-24 - rnn_length : -12]
    x_test = x[-12 - rnn_length:]

    y_train = y[: -24]
    y_eval = y[-24 - rnn_length : -12]
    y_test = y[-12 - rnn_length:]

    x_sentiment_train = x_sentiment[: -24]
    x_sentiment_eval = x_sentiment[-24 - rnn_length : -12]
    x_sentiment_test = x_sentiment[-12 - rnn_length:]

    # initialize
    best_model_file = 0
    epoch = 0
    wait_epoch = 0
    eval_epoch_best = 0

    model = AD_GAT(num_stock=NUM_STOCK, d_market = D_MARKET,d_news= D_NEWS,
                      d_hidden = D_MARKET, hidn_rnn = hidn_rnn, heads_att = heads_att,
                      hidn_att= hidn_att, dropout = args.dropout,t_mix = t_mix,
                      infer = infer, relation_static = static)
    # model = GCN(num_stock=NUM_STOCK, d_market = D_MARKET,d_news= D_NEWS,
    #                   d_hidden = D_MARKET, hidn_rnn = hidn_rnn, heads_att = heads_att,
    #                   hidn_att= hidn_att, dropout = args.dropout,t_mix = t_mix,
    #                   infer = infer, relation_static = static)

    model.cuda(device=DEVICE)
    model.to(torch.double)
    optimizer = optim.Adam(model.parameters(), lr= args.lr, weight_decay=args.weight_constraint)
    #train
    while epoch < MAX_EPOCH:
        train_loss = train(model, x_train,x_sentiment_train, y_train, relation_static = relation_static)
        eval_acc, eval_auc = evaluate(model, x_eval, x_sentiment_eval, y_eval, relation_static = relation_static)
        test_acc, test_auc = evaluate(model, x_test, x_sentiment_test, y_test, relation_static = relation_static)
        eval_str = "epoch{}, train_loss{:.4f}, eval_auc{:.4f}, eval_acc{:.4f}, test_auc{:.4f},test_acc{:.4f}".format(epoch, train_loss, eval_auc, eval_acc, test_auc, test_acc)
        print(eval_str)

        if eval_auc > eval_epoch_best:
            eval_epoch_best = eval_auc
            eval_best_str = "epoch{}, train_loss{:.4f}, eval_auc{:.4f}, eval_acc{:.4f}, test_auc{:.4f},test_acc{:.4f}".format(epoch, train_loss, eval_auc,eval_acc, test_auc, test_acc)
            wait_epoch = 0
            if args.save:
                if best_model_file:
                    os.remove(best_model_file)
                best_model_file = "./SavedModels/eval:auc{}_acc{}_test:auc{}_acc{}".format(eval_auc, eval_acc, test_auc, test_acc)
                torch.save(model.state_dict(), best_model_file)
        else:
            wait_epoch += 1

        if wait_epoch > 50:
            print("saved_model_result:",eval_best_str)
            break
        epoch += 1
