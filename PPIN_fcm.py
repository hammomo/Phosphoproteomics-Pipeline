#/usr/bin/python
# -*- coding:utf-8 -*-

import requests
import numpy as np
import pandas as pd
import os
import skfuzzy as fuzz
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import collections
import re
import time

string_api_url = "https://string-db.org/api"
output_format = "tsv-no-header"

def request_edges(gene_set, thres = 0.4):
    '''
    request edges from STRING
    '''
    method = "network"
    request_url = "/".join([string_api_url, output_format, method])
    params = {
        "identifiers" : "%0d".join(gene_set),
        "species" : 9606,
        "caller_identity" : "www.hanyi.mo.org"
    }
    response = requests.post(request_url, data=params)
    results_2d_arr = np.array([line.split('\t') for line in response.text.strip().split("\n")])
    if results_2d_arr.shape[1] < 11:
        return [], {}, len(gene_set), []
    valid_conns = np.array(list(filter(lambda x: float(x[10]) >= thres, results_2d_arr)))
    degrees = []
    conns = []
    conns_string_id = []
    gene_degree_map = {}
    if len(valid_conns):
        genes = np.unique(np.append(valid_conns[:, 2], valid_conns[:, 3]))
        for gene in genes:
            degree = len(list(filter(lambda x: x[2] == gene or x[3] == gene, list(valid_conns))))
            degrees.append(degree)
        gene_degree_map = dict(zip(list(genes), degrees))       
        for line in valid_conns:
            conns.append(tuple([line[2], line[3]]))
            conns_string_id.append(tuple([line[0], line[1]]))
    return conns, gene_degree_map, len(gene_set), conns_string_id

def calc_avg_cc(conns, gene_degree_map, local = True):
    '''
    calculate average clustering coefficient
    '''
    total_cc = 0
    n = len(gene_degree_map)
    if n == 0:
        return 0, 0, 0
    for gene in gene_degree_map.keys():
        degree = gene_degree_map[gene]
        if local:
            if degree == 1:
                total_cc += 1
                continue
        conn_single_gene = list(filter(lambda line: gene in line, conns))
        conn_genes = [[x for x in line if x != gene][0] for line in conn_single_gene]
        ni = sum([1 if conn[0] in conn_genes and conn[1] in conn_genes else 0 for conn in conns])
        if degree > 1: total_cc += 2 * ni / (degree * (degree - 1))
    return total_cc/n, total_cc, n

def fuzzy_c_means(data, ncentres, m, error = 0.005, maxiter=1000, seed = 1):
    '''
    run FCM
    '''
    cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
        data.transpose(), ncentres, m, error = error, maxiter = maxiter, seed = seed)
    cl = np.argmax(u, axis=0)
    max_membership = np.max(u, axis = 0)
    return cl, max_membership, cntr, u, fpc

def run_fcm_cc(data, start, end, m = 2):
    '''
    run FCM and calculate clustering coefficients for each number of centres
    '''
    data_tmp = data.copy()
    local_cc_mus = []
    for ncentres in range(start, end):
        start_time = time.time()
        cols = list(filter(lambda x: re.match(r'[0-9]*`', x), data.columns))
        cl, membership, cntr, u, fpc = fuzzy_c_means(data_tmp[cols].values, ncentres, m)
        data_tmp['cl'] = cl
        data_tmp['membership'] = membership
        cl_local_cc = {}
        for i in range(ncentres):
            gene_set = set(data_tmp[data_tmp['cl'] == i]['gene.symbol'])
            edges, degree_map, N, _ = request_edges(gene_set)
            avg_local_cc, local_cc, N = calc_avg_cc(edges, degree_map, False)
            cl_local_cc[i] = avg_local_cc
        cluster_mu = sum(cl_local_cc.values()) / ncentres
        local_cc_mus.append(cluster_mu)
        end_time = time.time()
        print('{0}s elapsed for {1} centres: average adjusted clustering coefficient per cluster: {2}\n'
          .format(end_time - start_time, ncentres, cluster_mu))
    return local_cc_mus

def plot_time_series(data, re_cols, cl_col, order, col_wrap = 2):
    '''
    plot temporal dynamics for each cluster
    '''
    counts = data.groupby(cl_col)[cl_col].transform('count')
    tmp = data.copy()
    tmp['cluster (n)'] = ['{0} ({1})'.format(x, y) for (x, y) in zip(tmp[cl_col], counts)]
    re_cols.append('cluster (n)')
    tmp = tmp.sort_values(by=[cl_col])
    order_dict = dict(zip(order, range(len(order))))
    plot_data = tmp.melt(id_vars = re_cols, var_name = 'time', value_name = 'Log2 (SILAC-ratio)')
    plot_data['Stimulation time [min]'] = [order_dict[time] for time in plot_data['time']]
    plt.rcParams['xtick.bottom'] = plt.rcParams['xtick.labelbottom'] = True
    plt.rcParams['xtick.top'] = plt.rcParams['xtick.labeltop'] = False
    sns.set(font_scale = 1.5)
    norm = plt.Normalize(np.min(plot_data['membership']), np.max(plot_data['membership']))
    sm = plt.cm.ScalarMappable(cmap="YlGnBu", norm=norm)
    sm.set_array([])
    ax = sns.relplot(x = 'Stimulation time [min]', y = 'Log2 (SILAC-ratio)', hue = 'membership', col='cluster (n)', col_wrap=col_wrap,
            kind = 'line', data= plot_data, legend=False, palette = 'YlGnBu')
    ax.set_xticklabels([''] + order)
    return sm

def calc_fuzzifier(D, N):
    '''
    calculate the optimal fuzzification parameter
    '''
    m = 1 + (1480/N + 22.05) * D ** (-2) + (12.33/N + 0.243) * D ** (-0.0406 * np.log(N) - 0.1134)
    return int(m)