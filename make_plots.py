'''
this script shall plot stuff out of csv in plots/plots_data folder
'''

import os

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
    plt.plot(df['mean_line'])
    plt.plot(df['up_line'])
    plt.plot(df['low_line'])
    plt.title(metadata_str)
    plt.show()