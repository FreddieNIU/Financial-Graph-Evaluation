import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from os import listdir
from matplotlib import ticker
import scipy
import random
import itertools
import sys 
sys.path.append("..")
from Generate_dataset import MyDataset
from utils import VisualizeGraphDataset, Describe, generate_tensor_list
from utils_fgim import *
import json
import statsmodels.api as sm
from tqdm import tqdm
import argparse
# ERGATFOLDERPATH = '/home/yingjie/Year_1/ERGAT/'
ERGATFOLDERPATH = '/home/yingjie_niu/Year_2/ERGAT-master/'
LOOKBACK = 22
SEED = -0.319

parser = argparse.ArgumentParser()

parser.add_argument('--dataset_fp',type=str, default='No',
                    help='path of the model file to be saved')

if __name__=="__main__":
    args = parser.parse_args()
    ## Load graph dataset
    dataset_fp = ERGATFOLDERPATH + args.dataset_fp
    dataset = MyDataset(root=dataset_fp)
    ReturnMatrix = pd.read_csv(ERGATFOLDERPATH + 'data/SP500/dataset202209-202310/ReturnMatrix.csv', index_col='Date')
    with open(ERGATFOLDERPATH+"data/202209_202310/Comp_To_Node_Idx.json",'r') as f:
        Comp_To_Node_Idx = json.load(f)
    Node_Idx_To_Comp = list(Comp_To_Node_Idx.keys())
    start, end = sorted(listdir(dataset_fp+'/processed'))[0].split('_')[1], sorted(listdir(dataset_fp+'/processed'))[-3].split('_')[1]
    print(f'\nDataset: {dataset_fp}')
    print(f"Dataset period from {start} to {end}")
    ReturnMatrix = ReturnMatrix.loc[start:end]

    # ## Return Correlation Stability Score (CSS) --------------------------------------------------------------------------------------------------
    # print('Calculating Correlation Stability Score ... ')
    # CSS = []
    # for day_t in tqdm(range(LOOKBACK, len(dataset)- LOOKBACK)):
    #     G_t, G_t_node_idx = get_connected_subgraph(dataset[day_t])
    #     connectedCorrDiff = ConnectedCorrDiff(G_t, day_t, Comp_To_Node_Idx, ReturnMatrix)
    #     isolatedCorrDiff = IsolatedCorrDiff(G_t,day_t, G_t_node_idx, Comp_To_Node_Idx, ReturnMatrix)
    #     ttest = scipy.stats.ttest_ind(connectedCorrDiff, isolatedCorrDiff, equal_var=False, alternative="greater")
    #     CS_t = 1 if ttest.pvalue < 0.01 else 0
    #     CSS.append(CS_t)
    # CSS = np.mean(CSS)
    # print(f"CSS: {CSS}")

    # ## Average Event Capture Rate (AECR)   --------------------------------------------------------------------------------------------------
    # print("Calculating Event Capture Rate ...")
    # # Iterate through all possible company pairs that have at least one edge in the dataset
    # NodePairsHaveEdge = torch.tensor([])
    # for i in range(len(dataset)):
    #     g = dataset[i]
    #     g_edges = g.edge_index.T
    #     if NodePairsHaveEdge.shape[0]==0:
    #         NodePairsHaveEdge = g_edges
    #     else:
    #         NodePairsHaveEdge = torch.concat([NodePairsHaveEdge, g_edges], dim=0)
    #     NodePairsHaveEdge= torch.unique(NodePairsHaveEdge, dim=0)

    # AECR = []
    # for idx in tqdm(range(NodePairsHaveEdge.shape[0])):
    #     comp1, comp2 = Node_Idx_To_Comp[NodePairsHaveEdge[idx][0]], Node_Idx_To_Comp[NodePairsHaveEdge[idx][1]]
    #     corr, date = historicalCorrelation(ReturnMatrix, comp1, comp2)
    #     edgeList = getEdgeList(dataset, comp1, comp2, Comp_To_Node_Idx)
    #     ECR = DetectDrops(corr, edgeList)
    #     AECR.append(ECR)
    # AECR = np.mean(AECR)
    # print(f"AECR: {AECR}")

    ## Delta-Beta   --------------------------------------------------------------------------------------------------
    ## Construct HML_R
    print("Calculating Delta_Beta ...")
    comp_list = list(Comp_To_Node_Idx.keys())
    comp_pair_list= list(itertools.combinations(comp_list, 2))
    comp_pair_edges_dict = {}
    for day in tqdm(range(LOOKBACK, len(dataset))):
        g = dataset[day]
        adj_matrix = EdgeIndex_To_AdjMatrix(graph=g, Comp_To_Node_Idx=Comp_To_Node_Idx)
        for pair in comp_pair_list:
            if pair not in comp_pair_edges_dict:
                comp_pair_edges_dict[pair] = []
            try:
                edge = adj_matrix[pair[0]].loc[pair[1]]
                comp_pair_edges_dict[pair].append(0) if np.isnan(edge) else comp_pair_edges_dict[pair].append(edge)
            except KeyError:
                comp_pair_edges_dict[pair].append(0)

    # from all companies pairs, find the pairs that has at least 1 edge
    non_zero_pairs = []
    non_zero_pair_edge = {}
    for pair in comp_pair_edges_dict:
        if np.count_nonzero(comp_pair_edges_dict[pair]) > 0:
            non_zero_pairs.append(pair)
            non_zero_pair_edge[pair] = np.count_nonzero(comp_pair_edges_dict[pair]) 
    # randomly sample 1200 company pairs from non zero pairs, and also save their corresponding number of edges
    random.seed(SEED)
    pairs_120 = random.sample(non_zero_pairs, 1200)
    edges_120 = []
    for pair in pairs_120:
        edges_120.append(non_zero_pair_edge[pair])
    plt.hist(edges_120, bins=10)
    # based on the number of edges, split the 120 companies into high-correlated, medium-correlated, low-correlated companies. 
    # low_point, high_point = 0.3*max(edges_120), 0.7*max(edges_120)
    low_point, high_point = min(edges_120) + 0.3*(max(edges_120)-min(edges_120)), min(edges_120) + 0.7*(max(edges_120)-min(edges_120))
    high_correlated_pairs = []
    medium_correlated_pairs = []
    low_correlated_pairs = []
    for pair in pairs_120:
        if non_zero_pair_edge[pair] > high_point:
            high_correlated_pairs.append(pair)
        elif non_zero_pair_edge[pair] > low_point:
            medium_correlated_pairs.append(pair)
        else:
            low_correlated_pairs.append(pair)
    # calculate the historical correlation of three group of companies
    # print(high_correlated_pairs, medium_correlated_pairs, low_correlated_pairs)
    high_corr = getCorrelationMeanOfCompanyPairs(high_correlated_pairs, ReturnMatrix)
    medium_corr = getCorrelationMeanOfCompanyPairs(medium_correlated_pairs, ReturnMatrix)
    low_corr = getCorrelationMeanOfCompanyPairs(low_correlated_pairs, ReturnMatrix)
    try:
        if np.isnan(high_corr):
            factor = low_corr
        else:   
            factor = high_corr - low_corr
    except ValueError:
        if np.isnan(high_corr).all():
            factor = low_corr
        else:   
            factor = high_corr - low_corr

    ##Test HML_R
    # randomly sample 100 company pairs from non zero pairs, and also save their corresponding number of edges
    random.seed(SEED)
    pairs_100 = random.sample(non_zero_pairs, 1000)
    edges_100 = {}
    for pair in pairs_100:
        edges_100[pair] = non_zero_pair_edge[pair]
    plt.hist(edges_100.values(), bins=10)
    # sorted the sampled 100 pairs by the number of edges
    sorted_pairs = sorted(edges_100.items(), key=lambda x:x[1])
    #split the pairs into 10 groups in the edge number ascending order
    sorted_10_group_corrs = {}
    for i in range(int(len(sorted_pairs)/100)):
        group = sorted_pairs[100*i:100*i+100]
        sorted_10_group_corrs[f'group_{i}'] = []
        for item in group:
            corr, _ = historicalCorrelation(ReturnMatrix, item[0][0], item[0][1])
            sorted_10_group_corrs[f'group_{i}'].append(corr)
        sorted_10_group_corrs[f'group_{i}'] = np.array(sorted_10_group_corrs[f'group_{i}'])
        sorted_10_group_corrs[f'group_{i}'] = np.mean(sorted_10_group_corrs[f'group_{i}'], axis=0)
    # regression 
    sm.add_constant(factor)
    betas = {}
    for key in sorted_10_group_corrs:
        mod = sm.OLS(sorted_10_group_corrs[key], factor)
        res = mod.fit()
        betas[key] = res.params[0]
    Delta_beta = np.mean(np.diff(list(betas.values())))
    print(f"Delta-Beta: {Delta_beta}")

    ## Delta-std   --------------------------------------------------------------------------------------------------
    high_diff = getReturnDiffMeanOfCompanyPairs(high_correlated_pairs, ReturnMatrix)
    medium_diff = getReturnDiffMeanOfCompanyPairs(medium_correlated_pairs, ReturnMatrix)
    low_diff = getReturnDiffMeanOfCompanyPairs(low_correlated_pairs, ReturnMatrix)

    Delta_std = np.std(low_diff) - np.std(high_diff)
    print(np.std(low_diff), np.std(high_diff))
    print(f"Delta-std: {Delta_std}")
 
    ## Delta-DCC   --------------------------------------------------------------------------------------------------
    high_ab = group_dcc_garch(high_correlated_pairs, ReturnMatrix)
    medium_ab = group_dcc_garch(medium_correlated_pairs, ReturnMatrix)
    low_ab = group_dcc_garch(low_correlated_pairs, ReturnMatrix)

    alpha_high = np.average(high_ab, axis=0)[0]
    beta_high = np.average(high_ab, axis=0)[1]
    alpha_low = np.average(low_ab, axis=0)[0]
    beta_low = np.average(low_ab, axis=0)[1]

    # Delta_DCC = alpha_high-alpha_low + beta_low - beta_high
    Delta_DCC = alpha_low-alpha_high + beta_high - beta_low

    print(f"Delta-DCC: {Delta_DCC} \n")


    # print(f"CSS: {CSS}, AECR: {AECR}, Delta-Beta: {Delta_beta}, Delta-std: {Delta_std}, Delta-DCC: {Delta_DCC} ")
    