#/usr/bin/python
# -*- coding:utf-8 -*-

import numpy as np
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm, chi2_contingency

def trim(f_name, time_series, col_map):
    '''
    trim the MaxQuant Phospho (STY).txt
    '''
    data = pd.read_csv(f_name , sep = '\t', low_memory=False)
    trimmed1 = data[(data['Potential contaminant'].isnull()) & (data['Reverse'].isnull())]
    trimmed2 = trimmed1[(trimmed1['Localization prob'] >= 0.75)]
    sty_cols_rep1 = list(filter(lambda x: re.match('Ratio [MH]/L normalized.*STY.*_01$', x), trimmed2.columns))
    sty_cols_rep2 = list(filter(lambda x: re.match('Ratio [MH]/L normalized.*STY.*_02$', x), trimmed2.columns))
    rep1 = trimmed2[list(sty_cols_rep1)].to_numpy()
    rep2 = trimmed2[list(sty_cols_rep2)].to_numpy()
    means = [(rep1[idx] + rep2[idx]) / 2 for idx in range(0, len(rep1))]
    cols = [col[:-3] for col in sty_cols_rep1]
    nums = pd.DataFrame(means, columns = cols, index = trimmed2.index)
    nums = nums.dropna(how='all')

    # all required columns for final table
    re_data = pd.concat([trimmed2[list(['Protein', 'Position', 'Amino acid', 'id', 'Protein names', 'Gene names',
                'Sequence window', 'Fasta headers'])], nums], axis=1, join='inner') # inner join

    # genes
    re_data['Gene names'] = re_data['Gene names'].apply(lambda x: x.split(';')[0] if isinstance(x, str) else np.nan)

    alt_genes = []
    for fa in re_data['Fasta headers']:
        m = re.search(r'GN=.*?[ ;]', fa)
        if m:
            alt_genes.append(m.group()[3:-1])
            print
        else:
            m = re.search(r'GN=.*?$', fa)
            alt_genes.append(m.group()[3:])
    re_data['gene'] = alt_genes
    
    # id concatenation
    new_ids = []
    for i in re_data.index:
        idx = re_data['id'][i]
        aa = re_data['Amino acid'][i]
        pos = 'NA' if np.isnan(re_data['Position'][i]) else str(int(re_data['Position'][i]))
        new_ids.append('{0}-{1}{2}'.format(idx, aa, pos))
    re_data['id'] = new_ids

    final = pd.DataFrame()
    final['id'] = new_ids
    final['protein'] = list(re_data['Protein'])
    final['protein.names'] = list(re_data['Protein names'])
    final['gene.symbol'] = [re_data['Gene names'][i] if isinstance(re_data['Gene names'][i], str)
                            else re_data['gene'][i] for i in re_data.index]
    # final['gene.symbol'] = list(re_data['Gene names'])
    final['amino.acid'] = list(re_data['Amino acid'])
    final['position'] = [round(float(x), 1) for x in list(re_data['Position'])]
    final['sequence'] = list(re_data['Sequence window'])
    q_cols = list(filter(lambda x: 'normalized' in x, re_data.columns))
    col_starts = col_map

    for s in col_starts:
        for t in time_series:
            final[s + '_' + t] = list(re_data[col_map[s] + '_' + t])
    final.index = np.arange(1, len(final) + 1)

    return final

def orga_new_data(data, conditions, time_indices):
    '''
    separate the trimmed data by stimulation
    '''
    re = {}
    for c in conditions:
        cols = list(filter(lambda x: x.startswith(c), data.columns))
        df = data[['id'] + cols].dropna().rename(columns=dict(zip(cols, time_indices)))
        re[c] = df
    return re

def transform(data, trans = 'log2'):
    '''
    transform data
    '''
    tmp = data.copy()
    if trans == 'log2':
        tmp.iloc[:, 1:] = tmp.iloc[:, 1:].apply(np.log2)
    elif trans == 'log10':
        tmp.iloc[:, 1:] = tmp.iloc[:, 1:].apply(np.log10)
    return tmp

def plot_dist(data, title, ax, show_thres = False):
    '''
    plot the distibution of SILAC ratios
    '''
    plot_data = data.melt(
            id_vars = ['id'], var_name = 'time', value_name = 'ratio')
    plt.subplot(ax)
    ax = sns.distplot(plot_data['ratio'], hist=True, kde=False, 
            bins=50)
    ax.set_title(title, fontsize = 20)
    if not show_thres: return None
    mu = np.mean(plot_data['ratio'])
    std = np.std(plot_data['ratio'])
    thres = [norm.ppf(0.05, loc = mu, scale = std), norm.ppf(0.95, loc = mu, scale = std)]
    plt.axvline(thres[0], 0,1, c='r')
    plt.axvline(thres[1], 0,1, c='r')
    return thres

def plot_corr_matrix(data, title, ax, i, cbar_ax):
    '''
    plot the Pearson's correlation matrix
    '''
    corr = data.corr()
    ax_i = sns.heatmap(
        corr,
        ax=ax,
        cbar=i == 0,
        vmin = 0, vmax = 1, #center = 0,
        cmap="YlGnBu",
        square = True,
        cbar_ax=None if i else cbar_ax,
        annot=True, fmt='.2f'
    )
    ax_i.set_title(title, fontsize = 20)

def identify_regulated(data, thres):
    '''
    filter regulated sites
    '''
    tmp = data.melt(
        id_vars = ['id'], var_name = 'time', value_name = 'ratio')
    ids = set(tmp[(tmp['ratio'] <= thres[0]) | (tmp['ratio'] >= thres[1])]['id'])
    re = data[data['id'].isin(ids)].copy()
    return re

def standardise(data):
    '''
    standardise to mean 0 and standard deviation 1
    '''
    re = []
    for row in data:
        re.append((row - np.mean(row))/np.std(row))
    return np.array(re)