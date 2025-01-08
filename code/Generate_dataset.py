import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data, Dataset, InMemoryDataset
from os import listdir
import os.path as osp
import os
from typing import List
import argparse
import pickle
from tqdm import tqdm
from utils import AutoUpdateQueue, merge_historical_graph
import json
import warnings
import itertools
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser()

parser.add_argument('--static', action="store_true",
                    help='default is False for dynamic graph, if want static graph, give "--static" ')
parser.add_argument('--num_node_features', type=int, default='6',
                    help='[Open, High, Low, Close, Volume, Dividends, Stock Splits]')
parser.add_argument('--num_edge_features', type=int, default='1',
                    help='1 Single Relationship, or other number for Multiple Relationships')
parser.add_argument('--num_Y_features', type=int, default='1',
                    help='1 Single Task, or other number for Multi Task')
parser.add_argument('--only_first_degree', type=bool, default=True,
                    help='True for only considering first-degree relationship')
parser.add_argument('--normalize', type=str, default='mean',
                    help='Node embedding normalization: "mean" for mean normalization, "min-max" for min-max normalization')
parser.add_argument('--threshold', type=float, default='0',
                    help='Threshold to filter related companies, default 0, no threshold')
parser.add_argument('--weighted_edge', type=str, default='No',
                    help='The way to assign weight to edges: "No": all edges have the same weight, "Counter": count the number of appearance of a company in one day, and use the number of appearance as the edge weight. "Correlation": use correlation as weight')
parser.add_argument('--task', type=int, default='0',
                    help='0 for graph dataset, 1 for ADGAT, ')
parser.add_argument('--build_edges', type=str, default='text',
                    help='How to build an edge? "text" means building an edge based on news, "thgnn" means building an edge using the method in thgnn, "merge" means merge current graph with in to the historical graph')
parser.add_argument('--forRNN', action="store_true",
                    help='Is this for GAT or RNN+GAT? default is False for GAT, if want for RNN+GAT, give "--forRNN"')
LOOKBACK_WINDOW = 12
LOOKBACK_GRAPH_Q = AutoUpdateQueue(maxsize=LOOKBACK_WINDOW)
# ERGATFOLDERPATH = '/home/yingjie/Year_1/ERGAT/'
ERGATFOLDERPATH = '/home/yingjie_niu/Year_2/ERGAT-master/'


def format_day(day:str) -> str:
    # Input: e.g. '2022-02-17', or '2022-10-07'
    # Output: e.g. '2022-2-17', or '2022-10-7'

    daySplit = day.split('-')
    for item in range(len(daySplit)):
        if daySplit[item][0]=='0':
            daySplit[item]=daySplit[item][1:]
    daySplit
    format_day = ''
    for item in daySplit:
        format_day += item + '-'
    format_day = format_day[:-1]
    return format_day

def format_related_company(newsData_ofCompany_onDay:np.array) -> List[str]:
    ListOfList = []
    for li in newsData_ofCompany_onDay:
        if isinstance(li, list):
            li = "'"+li[0]+"'"
        else:
            li = li[1:-1]
        companyInSingleQuotList = li.split(',')
        ListOfList.append(companyInSingleQuotList)
    return ListOfList

def check_title_row(df):
    """
    Checks whether the given DataFrame has a title row with the specified column names.
    If the title row is missing, adds it to the DataFrame.
    """
    expected_cols = ['Id', 'Company', 'Date', 'Title', 'Publisher', 'Link', 'ProviderPublishTime', 'Type', 'RelatedCompany', 'Text']
    
    if list(df.columns) == expected_cols:
        # Title row already exists, do nothing
        return df
    
    else:
        # Add title row to DataFrame
        df.columns = expected_cols
        df['RelatedCompany'][0] = [df.iloc[0]['Company']]
        df['Text'][0] = []
        return df

def Generate_Data_List(static=False, num_node_features=7, num_edge_features=1, num_Y_features=1, only_first_degree=True, normalization='mean', threshold=0, weighted_edge="No",build_edges="text", forRNN=False):
    '''
    dynamic: generate dynamic graph or static graph, default dynamic graph
    num_node_features: number of node embedding features
    num_edge_features: number of edge embedding features
    num_Y_features: number of label features
    only_first_degree: only consider first degree
    normalization: the way to normalize node embeddings,  e.g. "min-max", "mean"
    threshold: If --build_edges=="text", threshold to filter related companies, if the company appears more than threshold in the day, build an edge between this company and the source company. "threshold" should be an integer, e.g. 1
               If --build_edges=="thgnn", if the correlation between two companies is greater than "threshold" or less than "-threshold", then build an edge between them. "threshold" in this case should be a float between 0 and 1, e.g. 0.1
    weighted_edge: the way to assign weight to edges, e.g. "No": all edges have the same weight, "Counter": count the number of appearance of a company in one day, and use the number of appearance as the edge weight, "Correlation": use the correlation as the edge weight
    build_edges: the way to decide whether building an edge between two companies. "text" means building an edge based on news, "thgnn" means building an edge using the method in thgnn, "merge" means merge current graph with in to the historical graph'
    '''
    # %%

    folder_path =ERGATFOLDERPATH+f'data/SP500/dataset{PERIOD_START}-{PERIOD_END}/'
    # folder_path = ERGATFOLDERPATH
    # Load the company list
    company_list = np.load(folder_path + 'company_list.npy')
    
    # Load the return matrix
    ReturnMatrix = pd.read_csv(folder_path + 'ReturnMatrix.csv', index_col='Date')
    # Period of completed available data is 2022-02-17 --> 2022-08-19
    # ReturnMatrix = ReturnMatrix.loc['2022-02-17':'2022-08-19']
    # Period of completed available data is 2022-09-19 --> 2023-10-17
    if forRNN:
        ReturnMatrix = ReturnMatrix.loc['2022-10-06':'2023-10-17']
    else:
        ReturnMatrix = ReturnMatrix.loc['2022-09-20':'2023-10-17']



    # Find the news file and the quantitative file of company list, match the sequence
    news_with_related_company = folder_path + 'news_with_related_company/'
    quantitative_fp = folder_path + f'quantitative_{PERIOD_START}_{PERIOD_END}/'

    news_fileList = listdir(news_with_related_company)     # e.g. 'EBAY_news.csv'
    quantitative_fileList = listdir(quantitative_fp)      # e.g. 'EBAY.csv'
    news = []
    quantitative = []
    for company in company_list:
        if company+'_news.csv' in news_fileList:
            news.append(company+'_news.csv')
        else:
            print(company,'_news.csv not exist!')
        if company+'.csv' in quantitative_fileList:
            quantitative.append(company+'.csv')
        else:
            print(company,'.csv not exist!')
    # %%
    # For each day, create a dict to store the graph and corresponding Y on this day
    graphOfDays = {}

    # X --- load the quantitative data as node features
    # Edge_index --- load the co-occurrance companies in one news as the edges
    # Y --- 0 if the return of a company on the day is negative, otherwise 1 
    num_nodes = len(quantitative)
        
    # for idx in tqdm(range(ReturnMatrix.shape[0])):
    for idx in tqdm(range(ReturnMatrix.shape[0]-1)):
        day = ReturnMatrix.index[idx]
        nextday = ReturnMatrix.index[idx+1]
        # '2022-07-12 and 2022-07-13 do not have news and thus cannot generate a relational graph on those days
        if day == '2022-07-12' or day =='2022-07-13':
            pass
        else:
            if day not in graphOfDays:
                if only_first_degree:
                    # only consider first degree relationship
                    graphOfDays[day] = {
                        'X' : np.zeros((num_nodes,LOOKBACK_WINDOW+1,num_node_features)) if forRNN else np.zeros((num_nodes,num_node_features)),  # shape is [num_nodes, num_node_features]
                        'Edge_index_Transpose' : None,  # shape is [num_edges, 2], you should transpose before passing them to the data constructor
                        'Edge_attr' : np.zeros((num_nodes,num_edge_features)), # shape is [num_edge, num_edge_features]
                        'Y' : np.zeros((num_nodes, num_Y_features))  # node-level targets of shape [num_nodes, *]
                    }
                else:
                    pass
            Comp_To_Node_Idx = {}
            for node in range(num_nodes):
                company = quantitative[node].split('.')[0]
                Comp_To_Node_Idx[company] = node

            for node in range(num_nodes):
                company = quantitative[node].split('.')[0]
                # load quantitative data of the "company" on "day" as node features X
                if normalization == 'min-max':
                    # min-max normalization
                    quantiData_ofCompany = pd.read_csv(quantitative_fp+company+'.csv', index_col=0)
                    # quantiData_ofCompany = quantiData_ofCompany.reindex(columns=["Close","High","Low","Open","Volume","Dividends","Stock Splits"])
                    quantiData_ofCompany = quantiData_ofCompany.reindex(columns=["Adj Close","Close","High","Low","Open","Volume"])
                    normalized_quantiData_ofCompany = pd.DataFrame()

                    # normalize the DataFrame using mean normalization and replace columns with zero standard deviation with zeros
                    for col in quantiData_ofCompany.columns:
                        std_dev = quantiData_ofCompany[col].std()
                        if std_dev == 0:
                            # normalized_quantiData_ofCompany[col] = quantiData_ofCompany[col][0]
                            normalized_quantiData_ofCompany[col] = 0.5
                        else:
                            normalized_quantiData_ofCompany[col] = (quantiData_ofCompany[col] - quantiData_ofCompany[col].min()) / (quantiData_ofCompany[col].max() - quantiData_ofCompany[col].min())

                else:
                    # By default, mean normalization
                    quantiData_ofCompany = pd.read_csv(quantitative_fp+company+'.csv', index_col=0)
                    # quantiData_ofCompany = quantiData_ofCompany.reindex(columns=["Close","High","Low","Open","Volume","Dividends","Stock Splits"])
                    quantiData_ofCompany = quantiData_ofCompany.reindex(columns=["Adj Close","Close","High","Low","Open","Volume"])
                    
                    # normalized_quantiData_ofCompany = (quantiData_ofCompany-quantiData_ofCompany.mean())/quantiData_ofCompany.std()
                    # create a new DataFrame to store the normalized data
                    normalized_quantiData_ofCompany = pd.DataFrame()

                    # normalize the DataFrame using mean normalization and replace columns having zero standard deviations with zeros
                    for col in quantiData_ofCompany.columns:
                        std_dev = quantiData_ofCompany[col].std()
                        if std_dev == 0:
                            normalized_quantiData_ofCompany[col] = quantiData_ofCompany[col][0]
                        else:
                            normalized_quantiData_ofCompany[col] = (quantiData_ofCompany[col] - quantiData_ofCompany[col].mean()) / std_dev
                if not forRNN:
                    # node embedding "X" consists of the features of the "company" on "day"
                    # print(company,normalized_quantiData_ofCompany)
                    quantiData_ofCompany_onDay = normalized_quantiData_ofCompany.loc[day]
                    quantiData_ofCompany_onDay = quantiData_ofCompany_onDay.to_numpy()
                    graphOfDays[day]['X'][node] = quantiData_ofCompany_onDay[:num_node_features]
                else:
                    #node embedding "X" consists of the features of the "company" on "day-t" to "day", where t is the lookback window = 12
                    dayIndex = normalized_quantiData_ofCompany.index.get_loc(day)
                    quantiData_ofCompany_onDay = normalized_quantiData_ofCompany.iloc[dayIndex-LOOKBACK_WINDOW : dayIndex+1] # +1 to include the "day" itself
                    quantiData_ofCompany_onDay = quantiData_ofCompany_onDay[quantiData_ofCompany_onDay.columns[:num_node_features]].to_numpy()
                    graphOfDays[day]['X'][node] = quantiData_ofCompany_onDay
                    
                # generate the node level label Y, e.g. return classification task: 0 if return<0 and 1 if return>0.
                returnOfCompanyOnDay = ReturnMatrix[company].loc[nextday]
                graphOfDays[day]['Y'][node] = returnOfCompanyOnDay

                if build_edges=='text':
                    # load news data of the "company" on "day"
                    # if a in-list-company is mentioned with the "company" in the same news, build an edge between the two companies
                    newsData_ofCompany_onDay = pd.read_csv(news_with_related_company+company+'_news.csv')
                    newsData_ofCompany_onDay = check_title_row(newsData_ofCompany_onDay)
                    newsData_ofCompany_onDay = newsData_ofCompany_onDay[newsData_ofCompany_onDay['Date'] == format_day(day)]['RelatedCompany'].to_numpy()
                    companyInSingleQuotList = format_related_company(newsData_ofCompany_onDay)
                    Related_Comp_dict = {}
                    for li in companyInSingleQuotList:
                        # 'li': the list of mentioned companies in one news
                        if len(li) > 1:
                            li_ = []
                            for c in li:
                                if c != '':
                                    c = c.split("'")[1]
                                    if (c in company_list):
                                        li_.append(c)
                            comp_pairs = list(itertools.combinations(li_, 2))
                            for pair in comp_pairs:
                                if (pair not in Related_Comp_dict):
                                    Related_Comp_dict[pair] = 1
                                else:
                                    Related_Comp_dict[pair] += 1
                            for pair in Related_Comp_dict:   # 所有提到node的pair权重加一
                                if company in pair:
                                    Related_Comp_dict[pair] += 1
                        else:
                            for c in li:
                                if c != '':
                                    c = c.split("'")[1]
                                    if (c in company_list):
                                        pair = (company,c)
                                        if (pair not in Related_Comp_dict):
                                            Related_Comp_dict[pair] = 1
                                        else:
                                            Related_Comp_dict[pair] += 1
                        # for c in li:
                        #     if c != '':
                        #         c = c.split("'")[1]
                        #         if (c in company_list):
                        #             if (c not in Related_Comp_dict):
                        #                 Related_Comp_dict[c] = 1
                        #             else:
                        #                 Related_Comp_dict[c] += 1
                    for source_comp, target_comp in Related_Comp_dict:
                        # Do not consider self-relation,  
                        if source_comp != target_comp: 
                            if Related_Comp_dict[(source_comp, target_comp)] > threshold:
                                source = Comp_To_Node_Idx[source_comp]
                                target = Comp_To_Node_Idx[target_comp]
                                if graphOfDays[day]['Edge_index_Transpose'] is None:
                                    graphOfDays[day]['Edge_index_Transpose'] = np.array([[source, target],
                                                                                        [target, source]])  # undirectional graph
                                    if weighted_edge == 'No':
                                        graphOfDays[day]['Edge_attr'] = np.array([[1],
                                                                                            [1]]) # all edges have the same weight 1
                                    elif weighted_edge == 'Counter':
                                        graphOfDays[day]['Edge_attr'] = np.array([[Related_Comp_dict[(source_comp, target_comp)]],
                                                                                            [Related_Comp_dict[(source_comp, target_comp)]]]) # edges have different weight
                                else:
                                    graphOfDays[day]['Edge_index_Transpose'] = np.concatenate([graphOfDays[day]['Edge_index_Transpose'], np.array([[source, target], [target, source]])], axis=0)

                                    if weighted_edge == 'No':
                                        graphOfDays[day]['Edge_attr'] = np.concatenate([graphOfDays[day]['Edge_attr'], np.array([[1], [1]])], axis=0) # all edges have the same weight 1
                                    elif weighted_edge == 'Counter':
                                        graphOfDays[day]['Edge_attr'] = np.concatenate([graphOfDays[day]['Edge_attr'], np.array([[Related_Comp_dict[(source_comp, target_comp)]], [Related_Comp_dict[(source_comp, target_comp)]]])], axis=0) # edges have different weight
                    if static == True:
                        # print('static is True')
                        if day != '2022-10-06':
                            graphOfDays[day]['Edge_index_Transpose'] = graphOfDays['2022-10-06']['Edge_index_Transpose']
                            graphOfDays[day]['Edge_attr'] = graphOfDays['2022-10-06']['Edge_attr']
                
                elif build_edges=='thgnn':
                    thgnn_daily_relation = f'/home/yingjie/Year_1/THGNN-main/data/{PERIOD_START}_{PERIOD_END}/thgnn_daily_relation/'
                    relation_df = pd.read_csv(thgnn_daily_relation+day+'.csv',index_col=0)
                    relation_df = relation_df.reindex(company_list, columns=company_list)  # Delete the companies that are not in the list
                    
                    pos_idx = relation_df.index[relation_df[company]>threshold]
                    pos_comps = np.array(relation_df[company].loc[pos_idx].index)
                    pos_source_attr = np.array(relation_df[company].loc[pos_idx])
                    pos_source = [Comp_To_Node_Idx[comp] for comp in pos_comps]
                    pos_target = np.full_like(pos_source, Comp_To_Node_Idx[company])
                    pos_source_to_target = np.concatenate([[pos_source], [pos_target]], axis=0).T
                    pos_target_to_source = np.concatenate([[pos_target], [pos_source]], axis=0).T

                    neg_idx = relation_df.index[relation_df[company]<-threshold]
                    neg_comps = np.array(relation_df[company].loc[neg_idx].index)
                    neg_source_attr = np.array(relation_df[company].loc[neg_idx])
                    neg_source = [Comp_To_Node_Idx[comp] for comp in neg_comps]
                    neg_target = np.full_like(neg_source, Comp_To_Node_Idx[company])
                    neg_source_to_target = np.concatenate([[neg_source], [neg_target]], axis=0).T
                    neg_target_to_source = np.concatenate([[neg_target], [neg_source]], axis=0).T
                    
                    if graphOfDays[day]['Edge_index_Transpose'] is None:
                        graphOfDays[day]['Edge_index_Transpose'] = np.concatenate([np.concatenate([pos_source_to_target, pos_target_to_source], axis=0), np.concatenate([neg_source_to_target, neg_target_to_source],axis=0)], axis=0)
                        if weighted_edge == 'No':
                            graphOfDays[day]['Edge_attr'] = np.concatenate([np.ones_like(pos_source_attr),np.ones_like(pos_source_attr),-np.ones_like(neg_source_attr),-np.ones_like(neg_source_attr)], axis=0).reshape(-1,1)
                        elif weighted_edge == 'Correlation':
                            graphOfDays[day]['Edge_attr'] = np.concatenate([pos_source_attr,pos_source_attr,neg_source_attr,neg_source_attr], axis=0).reshape(-1,1)
                    else:
                        
                        if weighted_edge == 'No':
                            temp = {
                            'Edge_index_Transpose': np.concatenate([np.concatenate([pos_source_to_target, pos_target_to_source], axis=0), np.concatenate([neg_source_to_target, neg_target_to_source],axis=0)], axis=0),
                            'Edge_attr': np.concatenate([np.ones_like(pos_source_attr),np.ones_like(pos_source_attr),-np.ones_like(neg_source_attr),-np.ones_like(neg_source_attr)], axis=0).reshape(-1,1)
                        }
                            temp_merged_graph = merge_historical_graph(historical_graph=graphOfDays[day],graph_of_today=temp, forTHGNN=True)
                            graphOfDays[day]['Edge_index_Transpose'] = temp_merged_graph['Edge_index_Transpose']
                            graphOfDays[day]['Edge_attr'] = temp_merged_graph['Edge_attr']
                        elif weighted_edge == 'Correlation':
                            temp = {
                            'Edge_index_Transpose': np.concatenate([np.concatenate([pos_source_to_target, pos_target_to_source], axis=0), np.concatenate([neg_source_to_target, neg_target_to_source],axis=0)], axis=0),
                            'Edge_attr': np.concatenate([pos_source_attr,pos_source_attr,neg_source_attr,neg_source_attr], axis=0).reshape(-1,1)
                        }
                            temp_merged_graph = merge_historical_graph(historical_graph=graphOfDays[day],graph_of_today=temp, forTHGNN=True)
                            graphOfDays[day]['Edge_index_Transpose'] = temp_merged_graph['Edge_index_Transpose']
                            graphOfDays[day]['Edge_attr'] = temp_merged_graph['Edge_attr']
                    # for comp in relation_df.columns:
                    #     corr = relation_df[company].loc[comp]
                    #     if (corr > threshold or corr < -threshold) and comp != company:
                    #         target = Comp_To_Node_Idx[comp]
                    #         if graphOfDays[day]['Edge_index_Transpose'] is None:
                    #             graphOfDays[day]['Edge_index_Transpose'] = np.array([[source, target],
                    #                                                                 [target, source]])  # undirectional graph
                    #             if weighted_edge == 'No':
                    #                 if corr > threshold:
                    #                     graphOfDays[day]['Edge_attr'] = np.array([[1],
                    #                                                             [1]]) # all edges have the same weight 1
                    #                 else:
                    #                     graphOfDays[day]['Edge_attr'] = np.array([[-1],
                    #                                                             [-1]]) # all edges have the same weight 1
                    #             elif weighted_edge == 'Correlation':
                    #                 graphOfDays[day]['Edge_attr'] = np.array([[corr],
                    #                                                         [corr]]) # edges have different weight
                    #         elif np.any(np.all(graphOfDays[day]['Edge_index_Transpose'] == np.array([source, target]), axis=1)):
                    #                 existing_index1 = np.where(np.all(graphOfDays[day]['Edge_index_Transpose'] == np.array([source, target]), axis=1))[0][0]
                    #                 existing_index2 = np.where(np.all(graphOfDays[day]['Edge_index_Transpose'] == np.array([target, source]), axis=1))[0][0]
                    #                 if weighted_edge == 'Correlation':
                    #                     graphOfDays[day]['Edge_attr'][existing_index1] += corr
                    #                     graphOfDays[day]['Edge_attr'][existing_index2] += corr
                    #         else:
                    #             graphOfDays[day]['Edge_index_Transpose'] = np.concatenate([graphOfDays[day]['Edge_index_Transpose'], np.array([[source, target], [target, source]])], axis=0)

                    #             if weighted_edge == 'No':
                    #                 if corr > threshold:
                    #                     graphOfDays[day]['Edge_attr'] =np.concatenate([graphOfDays[day]['Edge_attr'], np.array([[1], [1]])], axis=0) # all edges have the same weight 1
                    #                 else:
                    #                     graphOfDays[day]['Edge_attr'] = np.concatenate([graphOfDays[day]['Edge_attr'], np.array([[-1], [-1]])], axis=0) # all edges have the same weight -1
                    #             elif weighted_edge == 'Correlation':
                    #                 graphOfDays[day]['Edge_attr'] = np.concatenate([graphOfDays[day]['Edge_attr'], np.array([[corr], [corr]])], axis=0) # edges have different weight
                
                elif build_edges=='merge':
                    # load news data of the "company" on "day"
                    # if a in-list-company is mentioned with the "company" in the same news, build an edge between the two companies on the graph of "day"
                    # merge the graph of day to the historical graph. The period of historical graph equals the lookback window
                    newsData_ofCompany_onDay = pd.read_csv(news_with_related_company+company+'_news.csv')
                    newsData_ofCompany_onDay = check_title_row(newsData_ofCompany_onDay)
                    newsData_ofCompany_onDay = newsData_ofCompany_onDay[newsData_ofCompany_onDay['Date'] == format_day(day)]['RelatedCompany'].to_numpy()
                    companyInSingleQuotList = format_related_company(newsData_ofCompany_onDay)
                    Related_Comp_dict = {}
                    for li in companyInSingleQuotList:
                        # 'li': the list of mentioned companies in one news
                        for c in li:
                            if c != '':
                                c = c.split("'")[1]
                                if (c in company_list):
                                    if (c not in Related_Comp_dict):
                                        Related_Comp_dict[c] = 1
                                    else:
                                        Related_Comp_dict[c] += 1
                    for comp in Related_Comp_dict:
                        # Do not consider self-relation,  
                        if comp != company: 
                            if Related_Comp_dict[comp] > threshold:
                                source = Comp_To_Node_Idx[company]
                                target = Comp_To_Node_Idx[comp]
                                if graphOfDays[day]['Edge_index_Transpose'] is None:
                                    graphOfDays[day]['Edge_index_Transpose'] = np.array([[source, target],
                                                                                        [target, source]])  # undirectional graph
                                    if weighted_edge == 'No':
                                        graphOfDays[day]['Edge_attr'] = np.array([[1],
                                                                                            [1]]) # all edges have the same weight 1
                                    elif weighted_edge == 'Counter':
                                        graphOfDays[day]['Edge_attr'] = np.array([[Related_Comp_dict[comp]],
                                                                                            [Related_Comp_dict[comp]]]) # edges have different weight
                                elif np.any(np.all(graphOfDays[day]['Edge_index_Transpose'] == np.array([source, target]), axis=1)):
                                    existing_index1 = np.where(np.all(graphOfDays[day]['Edge_index_Transpose'] == np.array([source, target]), axis=1))[0][0]
                                    existing_index2 = np.where(np.all(graphOfDays[day]['Edge_index_Transpose'] == np.array([target, source]), axis=1))[0][0]
                                    if weighted_edge == 'Counter':
                                        graphOfDays[day]['Edge_attr'][existing_index1] += Related_Comp_dict[comp] 
                                        graphOfDays[day]['Edge_attr'][existing_index2] += Related_Comp_dict[comp]
                                else:
                                    graphOfDays[day]['Edge_index_Transpose'] = np.concatenate([graphOfDays[day]['Edge_index_Transpose'], np.array([[source, target], [target, source]])], axis=0)

                                    if weighted_edge == 'No':
                                        graphOfDays[day]['Edge_attr'] = np.concatenate([graphOfDays[day]['Edge_attr'], np.array([[1], [1]])], axis=0) # all edges have the same weight 1
                                    elif weighted_edge == 'Counter':
                                        graphOfDays[day]['Edge_attr'] = np.concatenate([graphOfDays[day]['Edge_attr'], np.array([[Related_Comp_dict[comp]], [Related_Comp_dict[comp]]])], axis=0) # edges have different weight
            
            if graphOfDays[day]['Edge_index_Transpose'] is None:
                prev_day = ReturnMatrix.index[idx-1]
                # graphOfDays[day]['Edge_attr'] = graphOfDays[prev_day]['Edge_attr']
                # graphOfDays[day]['Edge_index_Transpose'] = graphOfDays[prev_day]['Edge_index_Transpose']
                prevday_graph = LOOKBACK_GRAPH_Q.get_last()
                graphOfDays[day]['Edge_attr'] = prevday_graph['Edge_attr']
                graphOfDays[day]['Edge_index_Transpose'] = prevday_graph['Edge_index_Transpose']
                print(f"NOTE: edge attr on {day} is empty, so the edge attr on {prev_day} is assigned to {day}")
            
            if normalization == 'min-max':
                std = graphOfDays[day]['Edge_attr'].std()
                if std == 0:
                    norm_graphOfDays_attr = np.ones_like(graphOfDays[day]['Edge_attr'])
                else:
                    norm_graphOfDays_attr = ( graphOfDays[day]['Edge_attr'] - graphOfDays[day]['Edge_attr'].min() ) / ( graphOfDays[day]['Edge_attr'].max() - graphOfDays[day]['Edge_attr'].min())
            else:
                std = graphOfDays[day]['Edge_attr'].std()
                if std == 0:
                    norm_graphOfDays_attr = np.ones_like(graphOfDays[day]['Edge_attr'])
                else:
                    norm_graphOfDays_attr = ( graphOfDays[day]['Edge_attr'] - graphOfDays[day]['Edge_attr'].mean() ) / std
            graphOfDays[day]['Edge_attr'] = norm_graphOfDays_attr
            
            temp = {
                'Edge_index_Transpose': graphOfDays[day]['Edge_index_Transpose'],
                'Edge_attr': graphOfDays[day]['Edge_attr']
            }
            LOOKBACK_GRAPH_Q.put(temp)
            # print(LOOKBACK_GRAPH_Q.get_last()['Edge_index_Transpose'].shape)
            if build_edges=="merge":
                merged_graph = None
                print("Historical length: ", len(LOOKBACK_GRAPH_Q.list))
                # for date in Q.list:   # 这里循环merge的方法有问题， 由于graphOfDays是在不断更新的， 上一天的graph已经是考虑lookback window merge过后的，再次循环merge不合理，可以考虑每次只merge当天与前一天的图（前一天的图是递归merge再前一天的图生成的以此类推）
                #     if merged_graph is None:
                #         merged_graph = graphOfDays[date]
                #     else:
                #         merged_graph = merge_historical_graph(merged_graph, graphOfDays[date])
                for graph in LOOKBACK_GRAPH_Q.list:
                    if merged_graph is None:
                        merged_graph = graph
                    else:
                        merged_graph = merge_historical_graph(historical_graph=merged_graph, graph_of_today=graph)
                # merged_graph = merge_historical_graph(merged_graph, graphOfDays[day])
                graphOfDays[day]['Edge_index_Transpose'] = merged_graph['Edge_index_Transpose']
                graphOfDays[day]['Edge_attr'] = merged_graph['Edge_attr']
            # print(LOOKBACK_GRAPH_Q.get_last()['Edge_index_Transpose'].shape)
            graphOfDays[day]['Edge_index_Transpose'] = graphOfDays[day]['Edge_index_Transpose'].astype(int)
    return graphOfDays

class MyDataset(Dataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super().__init__(root, transform, pre_transform)
        self.root = root
        # self.data, self.slices = torch.load(self.processed_paths[0])
    #返回数据集源文件名
    @property
    def raw_file_names(self):
        return 'ReturnMatrix.csv'
    #返回process方法所需的保存文件名。你之后保存的数据集名字和列表里的一致
    @property
    def processed_file_names(self):
        # return ['data_2022-08-18.pt']
        return ['data_2022-10-07.pt', 'data_2023-10-16.pt']
    #用于从网上下载数据集
    def download(self):
        # Download to `self.raw_dir`.
        pass
        
    #生成数据集所用的方法
    def process(self):
        # Read data into huge `Data` list.
        # # 这里用于构建data
        # Edge_index = torch.tensor([[0, 1, 1, 2],
        #                            [1, 0, 2, 1]], dtype=torch.long)

        # # 每个节点的特征：从0号节点开始。。
        # X = torch.tensor([[-1], [0], [1]], dtype=torch.float)
        # # 每个节点的标签：从0号节点开始-两类0，1
        # Y = torch.tensor([0,1,0],dtye=torch.float)


        # data = Data(x=x, edge_index=edge_index, y=Y)
        # # 放入datalist
        # data_list = [data]
        print('In Process')
        # data_list = []
        # print(args.static)
        graphOfDays = Generate_Data_List(
                        static=args.static,
                        num_node_features = args.num_node_features, 
                        num_edge_features = args.num_edge_features, 
                        num_Y_features = args.num_Y_features,
                        only_first_degree = args.only_first_degree,
                        threshold=args.threshold,
                        weighted_edge=args.weighted_edge,
                        build_edges=args.build_edges,
                        forRNN=args.forRNN
                        )
        for day in graphOfDays.keys():
            # print(day)
            # print(graphOfDays[day]['X'], graphOfDays[day]['Edge_index_Transpose'],graphOfDays[day]['Y'])
            data = Data(x=torch.tensor(graphOfDays[day]['X']), edge_index=torch.tensor(graphOfDays[day]['Edge_index_Transpose']).t(), edge_attr=torch.tensor(graphOfDays[day]['Edge_attr']), y=torch.tensor(graphOfDays[day]['Y']))
            
            torch.save(data, os.path.join(self.processed_dir, f'data_{day}.pt'))
        with open(self.root+'/args.json', 'wt') as f:
            json.dump(vars(args), f, indent=4)

    def len(self):
        namelist = listdir(self.root+'/processed')
        namelist.sort()
        namelist = namelist[:-2]
        return len(namelist)

    def get(self, idx):
        namelist = listdir(self.root+'/processed')
        namelist.sort()
        namelist = namelist[:-2]
        nl = [i.split('_')[1].split('.')[0] for i in namelist]
        data = torch.load(osp.join(self.processed_dir, f'data_{nl[idx]}.pt'))
        return data

def Genrate_Data_for_ADGAT(save_path:str, source_path:str):
    '''
    This method formulate our data to the format that ADGAT requires. The output
    consisits of four files: 

    x_numerical : array of shape [num_tradingdays, num_companies, num_numerical_features],  [*, 433, 7]
    y_ : array of shape [num_tradingdays, num_companies]  [*,433]
    x_textual : array of shape [num_tradingdays, num_companies, num_textual_features]  [*, 433, 1]
    relation : array of shape [num_companies, num_companies]  [433,433]
    '''
    print("\nGenerating dataset for ADGAT.")
    print('File saved in :', save_path,'\n')

    graphOfDays = Generate_Data_List(num_node_features = args.num_node_features, 
                        num_edge_features = args.num_edge_features, 
                        num_Y_features = args.num_Y_features,
                        only_first_degree = args.only_first_degree,
                        normalization=args.normalize,
                        forRNN=args.forRNN)
    
    x_numerical, y_, x_textual = [],[],[]
    for day in tqdm(graphOfDays.keys()):
        if day == '2022-07-12' or day =='2022-07-13':
            pass
        else:
            num_companies = graphOfDays[day]['X'].shape[0]

            x_numerical.append(graphOfDays[day]['X'])
            
            '''
            save textual data of day
            Since we don't have textual features, we set feature dimension to one, 
            and the value for all companies are the same, which is 1
            '''
            textualOfDay = np.ones((num_companies, 1))
            x_textual.append(textualOfDay)
            # save label of day
            y_.append(graphOfDays[day]['Y'].reshape((-1,)))
            # save the relation matrix of day
            relation = np.zeros((num_companies, num_companies))
            for pair in graphOfDays[day]['Edge_index_Transpose']:
                source, target = pair[0], pair[1]
                relation[source][target] = 1
            with open(save_path+'relations/relationOf{}.pkl'.format(day), 'wb') as handle:
                pickle.dump(relation, handle)
    x_numerical = np.stack(x_numerical, axis=0)
    x_textual = np.stack(x_textual,axis=0)
    y_ = np.stack(y_, axis=0)
    with open(save_path+'x_numerical.pkl', 'wb') as handle:
        pickle.dump(x_numerical, handle)
    with open(save_path+'x_textual.pkl', 'wb') as handle:
        pickle.dump(x_textual, handle)
    with open(save_path+'y_.pkl', 'wb') as handle:
        pickle.dump(y_, handle)
        


if __name__=="__main__":
    args = parser.parse_args()
    PERIOD_START = "202209"
    PERIOD_END = "202310"
    if args.task == 0:
        # For graph dataset
        # %%
        """测试"""
        # if args.static:
        #     b = MyDataset(root='../data/202202_202208/Graph_Dataset_'+args.normalize+'Norm_threshold'+str(args.threshold)+'_'+args.weighted_edge+'WeightedEdge_static_forRNN')
        # else:
        #     if args.build_edges=='text':
        #         b = MyDataset(root='../data/202202_202208/Graph_Dataset_'+args.normalize+'Norm_threshold'+str(args.threshold)+'_'+args.weighted_edge+'WeightedEdge_dynamic_forRNN')
        #     elif args.build_edges =='thgnn':
        #         b = MyDataset(root='../data/202202_202208/Graph_Dataset_'+args.normalize+'Norm_threshold'+str(args.threshold)+'_'+args.weighted_edge+'WeightedEdge_dynamic_forRNN')
        #     else:
        #         b = MyDataset(root='../data/202202_202208/Graph_Dataset_'+args.normalize+'Norm_threshold'+str(args.threshold)+'_'+args.weighted_edge+'WeightedEdge_0.9'+args.build_edges+'_dynamic_forRNN')
        if args.static:
            b = MyDataset(root=f'../data/{PERIOD_START}_{PERIOD_END}/static/Node{args.num_node_features}_Edge{args.num_edge_features}_Norm{args.normalize}_Threshold{args.threshold}_WeightedEdge{args.weighted_edge}_{args.build_edges}_{args.forRNN}')
        else:
            b = MyDataset(root=f'../data/{PERIOD_START}_{PERIOD_END}/dynamic/Node{args.num_node_features}_Edge{args.num_edge_features}_Norm{args.normalize}_Threshold{args.threshold}_WeightedEdge{args.weighted_edge}_{args.build_edges}_{args.forRNN}_2')
        print(b)
        

    elif args.task == 1:
        # Generate dataset for ADGAT
        Genrate_Data_for_ADGAT(save_path=ERGATFOLDERPATH+f'data/ADGAT_Data/{PERIOD_START}_{PERIOD_END}/static_threshold1_NodeFeature3/', source_path = ERGATFOLDERPATH)
        # Genrate_Data_for_ADGAT(save_path='/home/yingjie_niu/Year_2/ERGAT-master/data/ADGAT_Data/202202_202208/static_threshold3_NodeFeature3/', source_path = '../')

# %%
# import pickle
# save_path=ERGATFOLDERPATH+'data/ADGAT_Data/'
# day = '2022-07-14'
# with open(save_path+'relations/relationOf{}.pkl'.format(day), 'rb') as handle:
#     relation = pickle.load(handle)
# print(relation[0])
# print(relation[1])

# %%
