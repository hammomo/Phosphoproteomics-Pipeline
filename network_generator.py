#/usr/bin/python
# -*- coding:utf-8 -*-

import pandas as pd
import numpy as np
from statsmodels.stats.proportion import proportions_ztest
import seaborn as sns
import matplotlib.pyplot as plt
import requests
import os
from PPIN_fcm import request_edges

string_api_url = "https://string-db.org/api"
output_format = "tsv-no-header"
actions = pd.read_csv('./STRING_archive/9606.protein.actions.v11.0.txt', sep = '\t')
actions = actions[(actions['action'].notna())]
scores = pd.read_csv('./STRING_archive/9606.protein.links.full.v11.0.txt', sep = ' ')
scores = scores[scores['experiments'] >= 400]
actions = actions.merge(scores[['protein1', 'protein2', 'experiments']], how = 'left',
              left_on=['item_id_a', 'item_id_b'], right_on=['protein1', 'protein2'])
actions = actions[actions['experiments'].notna()]

def calc_pval_tables(data, conditions, ncentres):
    '''
    calculate p-value tables for all conditions
    '''
    pval_tables = {}
    for c in conditions:
        tmp_data = data[data['condition'] == c]
        nob = sum(tmp_data['membership'])
        arr = []
        for i in range(ncentres):
            count_i = sum(tmp_data[tmp_data['cl'] == i]['membership'])
            tmp = []
            for j in range(ncentres):
                count_j = sum(tmp_data[tmp_data['cl'] == j]['membership'])
                counts = np.array([count_i, count_j])
                nobs = np.array([nob] * 2)
                stat, pval = proportions_ztest(counts, nobs, alternative = 'larger')
                tmp.append(pval)
            arr.append(tmp)
        pvals = pd.DataFrame(arr, columns = np.arange(ncentres), index = np.arange(ncentres))
        pval_tables[c] = pvals
    return pval_tables

def plot_pval_table(data, ax, i, cbar_ax, title):
    '''
    plot heatmap according to p-value table
    '''
    ax_i = sns.heatmap(data, ax = ax, cbar = i == 0, vmin = 0, vmax = 1, 
                    cbar_ax=None if i else cbar_ax,
                      annot=True, fmt='.2f')
    ax_i.set_title('({0}) {1}'.format(str(chr(ord(str(i)) + 49)), title), fontsize = 45)

def enrich_condition(data, pval_table):
    '''
    return site table containing only enriched clusters
    '''
    dets = pval_table.applymap(lambda x: 1 if x < 0.05 else 0)
    dets['sum'] = dets.sum(axis=1)
    trimmed = data[data['cl'].isin(dets[dets['sum'] >= 4].index)].copy()
    trimmed['amino.acid.position'] = trimmed['id'].apply(lambda x: x.split('-')[1])
    trimmed = trimmed[['gene.symbol', 'cl', 'amino.acid.position']].drop_duplicates()
    return trimmed

def map_action(edge_list):
    '''
    obtain action types from STRING files
    '''
    arr = []
    for edge in edge_list.values:
        action = actions[(actions['item_id_a'] == '9606.' + edge[2]) & (actions['item_id_b'] == '9606.' + edge[3]) |
                    (actions['item_id_b'] == '9606.' + edge[3]) & (actions['item_id_a'] == '9606.' + edge[2])]
        if action.shape[0] > 0:
            action = action[action['is_directional'] == 't']
            for interaction in action.values:
                if interaction[5] == 't':
                    arr.append([edge[0], edge[1], interaction[3]])
                else:
                    arr.append([edge[1], edge[0], interaction[3]])
    return pd.DataFrame(arr, columns = ['node1', 'node2', 'interaction'])

def pair_cl_net(data, condition, pred_action_map):
    '''
    generate and save node and edge files
    '''
    same_gene_dict = {}
    edge_dict = {}
    node_dict = {}
    for k, v in pred_action_map.items():
        pred_action = v
        cl_i = k[0]
        gene_set_i = set(data[data['cl'] == cl_i]['gene.symbol'])
        edges_i, degrees_i, N_i, _ = request_edges(gene_set_i)
        cl_j = k[1]
        gene_set_j = set(data[data['cl'] == cl_j]['gene.symbol'])
        edges_j, degrees_j, N_j, _ = request_edges(gene_set_j)

        same_genes = list(filter(lambda x: x in degrees_j.keys(), degrees_i.keys()))
        if len(same_genes) < 1: 
            print('Cluster {0} and Cluster {1} contain no same gene.\n'.format(cl_i, cl_j))
        else:
            print('Cluster {0} and Cluster {1} contain {2} same genes.\nGene list: {3}\n'.format(cl_i, cl_j,
                                                                                            len(same_genes),
                                                                                            ', '.join(same_genes)))
            same_gene_dict[r'{0}&{1}'.format(cl_i, cl_j)] = data[
                (data['gene.symbol'].isin(same_genes) & 
                 data['cl'].isin([cl_i, cl_j]))][['gene.symbol',
                                                            'cl',
                                                            'amino.acid.position']].drop_duplicates()

        tot_gene_set = set(list(degrees_i.keys()) + list(degrees_j.keys()))
        tot_edges, tot_degrees, _, edges_string_id = request_edges(tot_gene_set)
        if len(tot_edges) == 0: continue
        edge_table = pd.DataFrame(np.hstack((tot_edges, edges_string_id)), columns = ['node1', 'node2', 'node1_id', 'node2_id'])
        edge_action_table = edge_table.iloc[:, :2].copy()
        edge_action_table['interaction'] = ['interaction'] * edge_action_table.shape[0]
        mapped_action_table = map_action(edge_table)
        edge_action_table = pd.concat([edge_action_table, mapped_action_table], ignore_index = True).drop_duplicates()
        new_arr = []
        arr_gene_cls = []
        preds = []
        for row in edge_action_table.values:
            if (row[0] not in same_genes) and (row[1] not in same_genes):
                node1 = row[0] + '-' + ';'.join(data[(data['gene.symbol'] == row[0]) &
                                            (data['cl'].isin([cl_i, cl_j]))]['amino.acid.position'].values)
                node2 = row[1] + '-' + ';'.join(data[(data['gene.symbol'] == row[1]) &
                                            (data['cl'].isin([cl_i, cl_j]))]['amino.acid.position'].values)
                new_arr.append([node1, node2, row[2]])
                node1_cl = str(cl_i) if row[0] in degrees_i.keys() else str(cl_j)
                node2_cl = str(cl_i) if row[1] in degrees_i.keys() else str(cl_j)
                if node1_cl != node2_cl:
                    if set([node1, node2]) not in preds: preds.append(set([node1, node2]))
                arr_gene_cls.append([node1, node1_cl])
                arr_gene_cls.append([node2, node2_cl])
                continue
            if (row[0] in same_genes) and (row[1] not in same_genes):
                node2 = row[1] + '-' + ';'.join(data[(data['gene.symbol'] == row[1]) &
                                            (data['cl'].isin([cl_i, cl_j]))]['amino.acid.position'].values)
                node1i = row[0] + '-' + ';'.join(data[(data['gene.symbol'] == row[0]) &
                                            (data['cl'] == cl_i)]['amino.acid.position'].values)
                node1j = row[0] + '-' + ';'.join(data[(data['gene.symbol'] == row[0]) &
                                            (data['cl'] == cl_j)]['amino.acid.position'].values)
                new_arr.append([node1i, node2, row[2]])
                new_arr.append([node1j, node2, row[2]])
                new_arr.append([node1i, node1j, 'cleavage'])
                node2_cl = str(cl_i) if row[1] in degrees_i.keys() else str(cl_j)
                if node2_cl != str(cl_i):
                    if set([node1i, node2]) not in preds: preds.append(set([node1i, node2]))
                else:
                    if set([node1j, node2]) not in preds: preds.append(set([node1j, node2]))
                arr_gene_cls.append([node2, node2_cl])
                arr_gene_cls.append([node1i, str(cl_i)])
                arr_gene_cls.append([node1j, str(cl_j)])
                continue
            if (row[0] not in same_genes) and (row[1] in same_genes):
                node1 = row[0] + '-' + ';'.join(data[(data['gene.symbol'] == row[0]) &
                                            (data['cl'].isin([cl_i, cl_j]))]['amino.acid.position'].values)
                node2i = row[1] + '-' + ';'.join(data[(data['gene.symbol'] == row[1]) &
                                            (data['cl'] == cl_i)]['amino.acid.position'].values)
                node2j = row[1] + '-' + ';'.join(data[(data['gene.symbol'] == row[1]) &
                                            (data['cl'] == cl_j)]['amino.acid.position'].values)
                new_arr.append([node1, node2i, row[2]])
                new_arr.append([node1, node2j, row[2]])
                new_arr.append([node2i, node2j, 'cleavage'])
                node1_cl = str(cl_i) if row[0] in degrees_i.keys() else str(cl_j)
                if node1_cl != str(cl_i):
                    if set([node1, node2i]) not in preds: preds.append(set([node1, node2i]))
                else:
                    if set([node1, node2j]) not in preds:preds.append(set([node1, node2j]))
                arr_gene_cls.append([node1, node1_cl])
                arr_gene_cls.append([node2i, str(cl_i)])
                arr_gene_cls.append([node2j, str(cl_j)])
                continue
            if (row[0] in same_genes) and (row[1] in same_genes):
                node1i = row[0] + '-' + ';'.join(data[(data['gene.symbol'] == row[0]) &
                                            (data['cl'] == cl_i)]['amino.acid.position'].values)
                node1j = row[0] + '-' + ';'.join(data[(data['gene.symbol'] == row[0]) &
                                            (data['cl'] == cl_j)]['amino.acid.position'].values)
                node2i = row[1] + '-' + ';'.join(data[(data['gene.symbol'] == row[1]) &
                                            (data['cl'] == cl_i)]['amino.acid.position'].values)
                node2j = row[1] + '-' + ';'.join(data[(data['gene.symbol'] == row[1]) &
                                            (data['cl'] == cl_j)]['amino.acid.position'].values)
                new_arr.append([node1i, node2i, row[2]])
                new_arr.append([node1j, node2j, row[2]])
                new_arr.append([node1i, node1j, 'cleavage'])
                new_arr.append([node2i, node2j, 'cleavage'])
                if set([node1i, node2j]) not in preds: preds.append(set([node1i, node2j]))
                if set([node1j, node2i]) not in preds: preds.append(set([node1j, node2i]))

                arr_gene_cls.append([node1i, str(cl_i)])
                arr_gene_cls.append([node1j, str(cl_j)])
                arr_gene_cls.append([node2i, str(cl_i)])
                arr_gene_cls.append([node2j, str(cl_j)])
        node_table = pd.DataFrame(arr_gene_cls, columns = ['node', 'cl']).drop_duplicates()
        new_edge_action_table = pd.DataFrame(new_arr, columns = ['node1', 'node2', 'interaction']).drop_duplicates()
        preds_table = pd.DataFrame([list(s) for s in preds], columns = ['node1', 'node2']).drop_duplicates()
        preds_table['interaction'] = [pred_action] * preds_table.shape[0]
        edge_dict[r'{0}&{1}'.format(cl_i, cl_j)] = preds_table
        node_dict[r'{0}&{1}'.format(cl_i, cl_j)] = node_table
        new_edge_action_table = pd.concat([new_edge_action_table, preds_table], ignore_index = True)

        folder_name = './conditions/{0}'.format(condition)
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
        new_edge_action_table.to_csv('{0}/edges_cl{1}&cl{2}.csv'.format(folder_name, cl_i, cl_j), index = None)
        node_table.to_csv('{0}/nodes_cl{1}&cl{2}.csv'.format(folder_name, cl_i, cl_j), index = None)
    return same_gene_dict, edge_dict, node_dict