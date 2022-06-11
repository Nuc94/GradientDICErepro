'''
this script shall plot stuff out of csv in plots/plots_data folder
'''

import os
import json

import numpy as np
import pandas as pd

from matplotlib import pyplot as plt

PLOTS_DATA_PATH = 'plots/plots_data/'
PLOTS_DATA_PATH = os.path.join('plots','plots_data')

def plot_df(df, savepath):
    x = np.arange(0.0, len(df), 1)
    plt.plot(df['mean_line'], color = 'b')
    #plt.plot(df['up_line'])
    #plt.plot(df['low_line'])
    plt.fill_between(x, df['up_line'], df['mean_line'], alpha = 0.18, color = 'b')
    plt.fill_between(x, df['mean_line'], df['low_line'], alpha = 0.18, color = 'b')
    plt.title(metadata_str)
    plt.savefig(savepath)
    plt.clf()

data_files = os.listdir(PLOTS_DATA_PATH)

data_grouped = dict()

#and boh plotting everything one by one
for data_file in data_files:
    filepath = os.path.join(PLOTS_DATA_PATH, data_file)
    df = pd.read_csv(filepath)
    metadata_str = data_file[:-14]
    metadata = json.loads(metadata_str)
    game = metadata['game']
    algo = metadata['algo']
    discount = metadata['discount']
    lr = metadata['lr']
    ridge = metadata['ridge']
    if game not in data_grouped.keys():
        data_grouped[game] = dict()
    if discount not in data_grouped[game].keys():
        data_grouped[game][discount] = dict()
    if algo not in data_grouped[game][discount].keys():
        data_grouped[game][discount][algo] = list()
    data_grouped[game][discount][algo].append( {
        'metadata' : metadata,
        'metadata_str' : metadata_str,
        'df': df
    } )
    savepath = os.path.join(os.path.join('plots', 'plots'), metadata_str + '.png')
    plot_df(df, savepath)

#i shall then loop through the available plots to check the best ones
for game in data_grouped.keys():
    for discount in data_grouped[game].keys():
        for algo in data_grouped[game][discount].keys():
            best_perf = np.inf
            best_entry = None
            for entry in data_grouped[game][discount][algo]:
                perf = list(entry['df']['mean_line'])[-1]
                if perf < best_perf:
                    best_perf = perf
                    best_entry = entry
            metadata_str = best_entry['metadata_str']
            savepath = os.path.join(os.path.join(os.path.join('plots', 'plots'), 'plots_best'), metadata_str + '.png')
            df = best_entry['df']
            plot_df(df, savepath)