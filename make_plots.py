'''
this script shall plot stuff out of csv in plots/plots_data folder
'''

import os
from turtle import color

import numpy as np
import pandas as pd

from matplotlib import pyplot as plt

PLOTS_DATA_PATH = 'plots/plots_data/'
PLOTS_DATA_PATH = os.path.join('plots','plots_data')

data_files = os.listdir(PLOTS_DATA_PATH)

#and boh plotting everything one by one
for data_file in data_files:
    filepath = os.path.join(PLOTS_DATA_PATH, data_file)
    metadata_str = data_file[:-4]
    df = pd.read_csv(filepath)
    x = np.arange(0.0, len(df), 1)
    plt.plot(df['mean_line'], color = 'b')
    #plt.plot(df['up_line'])
    #plt.plot(df['low_line'])
    plt.fill_between(x, df['up_line'], df['mean_line'], alpha = 0.18, color = 'b')
    plt.fill_between(x, df['mean_line'], df['low_line'], alpha = 0.18, color = 'b')
    plt.title(metadata_str)
    plt.show()