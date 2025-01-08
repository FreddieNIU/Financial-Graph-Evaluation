import matplotlib.pyplot as plt
import scipy
import numpy as np
import pandas as pd
import torch
from DCC_GARCH.GARCH.GARCH import GARCH
from DCC_GARCH.GARCH.GARCH_loss import garch_loss_gen
from DCC_GARCH.DCC.DCC import DCC
from DCC_GARCH.DCC.DCC_loss import dcc_loss_gen
LOOKBACK = 22

def historicalCorrelation(ReturnMatrix, comp1, comp2):
    """
    Input: ReturnMatrix, the company tickers of the two company to calculate
    Output: the Pearson Correlation Coefficient of the two companies per lookback window
    """
    # date = []
    # corr = []
    # for i in range(ReturnMatrix.shape[0]-LOOKBACK):
    #     returnSample = ReturnMatrix[[comp1,comp2]].iloc[i:i+LOOKBACK]
    #     corr.append(np.corrcoef(returnSample, rowvar=False)[0,1])
    #     date.append(ReturnMatrix.index[i+LOOKBACK])
    # corr = np.array(corr)
    hist_corr = ReturnMatrix[comp1].rolling(window=LOOKBACK).corr(ReturnMatrix[comp2])
    hist_corr = hist_corr[LOOKBACK-1:]
    corr = hist_corr.values
    date = hist_corr.index
    return corr, date

def ConnectedCorrDiff(G_t, day_t, Comp_To_Node_Idx, ReturnMatrix):
    """
    Input: 
    G_t: Graph of day t
    day t: day t 
    Comp_To_Node_Idx : a dictionary from company to node index
    ReturnNatrix : return matrix
    
    Output: 
    A list of correlations difference of each pair of directly connected nodes in the G_t
    The correlation difference = the correlation after this edge were built - the correlation before this edge where built
    """
    # 自定义函数
    def custom_function(x):
        Node_Idx_To_Comp = list(Comp_To_Node_Idx.keys())
        a = lambda x: [Node_Idx_To_Comp[x[0]], Node_Idx_To_Comp[x[1]]]
        node1 = a(x)[0]
        node2 = a(x)[1]
        hist_corr = ReturnMatrix[node1].rolling(window=LOOKBACK).corr(ReturnMatrix[node2])
        # hist_corr, _ = historicalCorrelation(ReturnMatrix, node1, node2)
        prev_corr, post_corr = hist_corr[day_t - 1], hist_corr[day_t + LOOKBACK-1]

        return abs(post_corr-prev_corr)

    # 将相互链接的节点对的edge_index赋值为一个二维数组
    original_array = np.array(G_t.edge_index)

    # 对每一列应用自定义函数
    corr_diff_li = np.apply_along_axis(custom_function, axis=0, arr=original_array)
            
    return corr_diff_li

def IsolatedCorrDiff(G_t,day_t, G_t_node_idx, Comp_To_Node_Idx, ReturnMatrix):
    """
    Input:
    Output: 
    """
    Node_Idx_To_Comp = list(Comp_To_Node_Idx.keys())
    node_idx = torch.tensor(list(Comp_To_Node_Idx.values()))
    isolated_node_idx = node_idx[~torch.isin(node_idx, G_t_node_idx)]
    # iterate all edges and calculate the correlation of node pairs of each edge
    # store the correlations difference of each pair of connected nodes in corr_diff_li
    corr_diff_li = []
    pairs = []
    for i in range(G_t.edge_index.shape[1]):
        if isolated_node_idx.shape[0] == 0:
            corr_diff_li.append(0)
        else:
            random_index = torch.randint(0, isolated_node_idx.shape[0], (1,2))[0]
            random_index_set = set(np.array(random_index))
            if random_index_set not in pairs:
                pairs.append(random_index_set)
                node1 = Node_Idx_To_Comp[isolated_node_idx[random_index[0]]]
                node2 = Node_Idx_To_Comp[isolated_node_idx[random_index[1]]]
                hist_corr = ReturnMatrix[node1].rolling(window=LOOKBACK).corr(ReturnMatrix[node2])
                # hist_corr, _ = historicalCorrelation(ReturnMatrix, node1, node2)
                prev_corr, post_corr = hist_corr[day_t - 1], hist_corr[day_t + LOOKBACK-1]
                corr_diff_li.append(post_corr-prev_corr)
    return corr_diff_li 

def getEdgeList(dataset, comp1, comp2, Comp_To_Node_Idx):
    """
    Return the edges between comp1 and comp2 as a list. If there is an edge, append the edge attribute, elses append 0. 
    """
    edge_list = []
    idx1, idx2 = Comp_To_Node_Idx[comp1], Comp_To_Node_Idx[comp2]
    for day in range(LOOKBACK, len(dataset)):
        g = dataset[day]

        # adj_matrix = EdgeIndex_To_AdjMatrix(graph=g, Comp_To_Node_Idx=Comp_To_Node_Idx)
        try:
            # edge = adj_matrix[comp1].loc[comp2]
            edge = g.edge_attr[np.logical_and(g.edge_index[0]==idx1, g.edge_index[1]==idx2) == 1]
            if edge.shape[0] != 0:
                edge = edge[-1].item()
            else:
                edge = np.nan
            edge_list.append(0) if np.isnan(edge) else edge_list.append(edge)
        except KeyError:
            edge_list.append(0)
    return edge_list

def DetectDrops(corr, edges, event_window=3):
    """
    Given the correlation series of two companies in the whole dataset period, and the edges built in the same period, this
    fnction returns the correlation maximum drop during the period, and the drop in the event period. 
    The event period is defined as follow:
        if there is an edge on day 't', event period = t-3 --> t+3
        if there are edges from day 't' to day 't+5', event period = 't-3' --> 't+8'
    This function also returns the rate between the drop in the event period and the max drop.
    This function also compairs the drop in the event period and the std in the whole period.
    """
    std = np.std(corr)
    max_drop = max(corr) - min(corr)

    event_periods_starts = []
    event_periods_ends = []
    x_prev = 0
    for i, x in enumerate(edges):
        if x != 0 and x_prev == 0:
            event_periods_starts.append(i)
        if x == 0 and x_prev != 0:
            event_periods_ends.append(i)
        x_prev = x
    if len(event_periods_starts) != len(event_periods_ends):
        event_periods_ends.append(len(edges)+1)
    if len(event_periods_starts) != len(event_periods_ends):
        raise IndexError(f'event_period_starts and event_period_ends length mis-match. {len(event_periods_starts), len(event_periods_ends)}')
    else:
        event_drops = {}
        for idx in range(len(event_periods_starts)):
            start = event_periods_starts[idx] - event_window
            end = event_periods_ends[idx] + event_window
            event_period_corr = corr[start:end]
            if len(event_period_corr) == 0:
                event_drop = 0
            else:
                event_drop = max(event_period_corr) - min(event_period_corr)
            # event_drop = max(event_period_corr) - min(event_period_corr)
            event_drops[(start, end)] = event_drop
    # print(f"Max Drop: {max_drop:.2f}, Std: {std:.2f}")
    ECR = []
    for event_period in event_drops:
        event_drop = event_drops[event_period]
        # print(f"Event Drop: {event_drop:.2f}, Drop Rate: {event_drop/max_drop:.2f}")
        EC_Te = 1 if event_drop/max_drop > std else 0
        ECR.append(EC_Te)
    if len(ECR) == 0:
        ECR.append(0)
    return np.mean(ECR)

def EdgeIndex_To_AdjMatrix(graph, Comp_To_Node_Idx):
    """
    Input: graph as type torch_geometric.data.data.Data, Comp_To_Node_Idx as type dict
    Output: adjacency matrix graph
    """
    edge_index, edge_attr = graph.edge_index, graph.edge_attr
    nodes = torch.tensor(np.array(list(set(np.array(edge_index[0])))))
    Node_Idx_To_Comp = list(Comp_To_Node_Idx.keys())
    num_edges = edge_index.shape[1]
    adj_matrix = pd.DataFrame(columns=[Node_Idx_To_Comp[node] for node in nodes], index=[Node_Idx_To_Comp[node] for node in nodes])
    for i in range(num_edges):
        source, target = edge_index[:,i]
        source = Node_Idx_To_Comp[source.item()]
        target = Node_Idx_To_Comp[target.item()]
        attr = edge_attr[i].item()
        adj_matrix[source].loc[target]=attr
        
    return adj_matrix

def getCorrelationMeanOfCompanyPairs(correlated_pairs, ReturnMatrix):
    '''
    Input: 
    '''
    correlation_ts = []
    for pair in correlated_pairs:
        corr, _ = historicalCorrelation(ReturnMatrix, pair[0], pair[1])
        correlation_ts.append(corr)
    correlation_ts = np.array(correlation_ts)
    correlaion_mean_ts = np.mean(correlation_ts, axis=0)
    return correlaion_mean_ts

# calculate the historical return_difference of three group of companies
def historicalReturnDiffDiff(ReturnMatrix, comp1, comp2):
    """
    Input: ReturnMatrix, the company tickers of the two company to calculate
    Output: the return difference of the two companies
    """
    date = []
    diff = []
    for i in range(1,ReturnMatrix.shape[0]-LOOKBACK):
        # diff.append(ReturnMatrix[comp1].iloc[i] - ReturnMatrix[comp1].iloc[i-1] - (ReturnMatrix[comp2].iloc[i] - ReturnMatrix[comp2].iloc[i-1]))
        # diff.append(min(ReturnMatrix[comp1].iloc[i]-ReturnMatrix[comp2].iloc[i],abs(ReturnMatrix[comp1].iloc[i])-abs(ReturnMatrix[comp2].iloc[i])))
        diff.append(abs(abs(ReturnMatrix[comp1].iloc[i])-abs(ReturnMatrix[comp2].iloc[i])))
        date.append(ReturnMatrix.index[i+LOOKBACK])
    diff = np.array(diff)
    return diff, date

def getReturnDiffMeanOfCompanyPairs(correlated_pairs, ReturnMatrix):
    diff_ts = []
    for pair in correlated_pairs:
        diff, _ = historicalReturnDiffDiff(ReturnMatrix, pair[0], pair[1])
        diff_ts.append(diff)
    diff_ts = np.array(diff_ts)   
    # diff_mean_ts = np.mean(diff_ts, axis=0)
    return diff_ts

def dcc_garch(comp1, comp2, ReturnMatrix):
    """
    Given a pair of companies, return the coefficient a, b of DCC GARCH model.
    """

    c1_garch_model = GARCH(1,1)
    c1_garch_model.set_loss(garch_loss_gen(1,1))
    c1_garch_model.set_max_itr(1)
    c1_return = ReturnMatrix[comp1].iloc[::-1]
    c1_garch_model.fit(c1_return)

    c2_garch_model = GARCH(1,1)
    c2_garch_model.set_loss(garch_loss_gen(1,1))
    c2_garch_model.set_max_itr(1)
    c2_return = ReturnMatrix[comp2].iloc[::-1]
    c2_garch_model.fit(c2_return)

    c1_sigma = c1_garch_model.sigma(c1_return)
    c1_epsilon = c1_return / c1_sigma

    c2_sigma = c2_garch_model.sigma(c2_return)
    c2_epsilon = c2_return / c2_sigma

    epsilon = np.array([c1_epsilon, c2_epsilon])

    dcc_model = DCC()
    dcc_model.set_loss(dcc_loss_gen())
    dcc_model.fit(epsilon)

    return dcc_model.get_ab()

def group_dcc_garch(group, ReturnMatrix):
    """
    Input: group of pairs
    Output: a, b of each pair
    """
    group_ab= []
    for pair in group:
        try: 
            ab = dcc_garch(pair[0],pair[1], ReturnMatrix)
        except np.linalg.LinAlgError:
            pass
        group_ab.append(ab)
    
    return group_ab

def get_connected_subgraph(data):
    """
    Input is a pyg.Data
    Output is a subgraph containing only the nodes whose degree > 1
    """
    connected_nodes = torch.tensor(np.array(list(set(np.array(data.edge_index[0])))))
    connected_subgraph = data.subgraph(connected_nodes)
    return connected_subgraph, connected_nodes