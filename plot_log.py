'''
ok here i hope to effectively check the logs available
'''

import os
import json
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tensorflow.python.summary.summary_iterator import summary_iterator

def metadataFromLogDirName(log_dir):
    '''
    this dumb method shall allow me to extract metadata from a logging
    directory, so as to check whether or not stuff stays there

    example of input string:
    logger-BoyansChainLinear-v0-activation_linear-algo_GradientDICE-discount_0.9-log_level_0-lr_0.25-ridge_0-run-0-220425-121302
    '''
    metadata = dict()
    if 'BoyansChainTabular-v0' in log_dir:
        metadata['game'] = 'BoyansChainTabular-v0'
    elif 'BoyansChainLinear-v0' in log_dir:
        metadata['game'] = 'BoyansChainLinear-v0'
    else:
        metadata['game'] = 'unkown'
    #then i shall get the algo
    algo_split = log_dir.split('algo_')[-1]
    dice_split = algo_split.split('DICE')[0]
    metadata['algo'] = dice_split + 'DICE'
    #then the discount factor
    discount_split = log_dir.split('discount_')[-1]
    discount = float(discount_split.split('-')[0])
    #then the learning rate
    lr_split = log_dir.split('-lr_')[-1]
    lr = float(lr_split.split('-')[0])
    #and ridge
    ridge_split = log_dir.split('-ridge_')[-1]
    ridge = float(ridge_split.split('-')[0])
    metadata['discount'] = discount
    metadata['lr'] = lr
    metadata['ridge'] = ridge
    return metadata

def metadataFromVars(game, algo, discount, lr, ridge):
    metadata = {
        'game' : game,
        'algo' : algo,
        'discount' : discount,
        'lr' : lr,
        'ridge' : ridge
    }
    return metadata

def metadataToString(metadata):
    return json.dumps(metadata).replace('\\', '')

ptf = '/home/nuc/Programmazione/Python/RLRepro/DeepRL/tf_log/logger-BoyansChainTabular-v0-activation_squared-algo_GenDICE-discount_0.1-lam_1-lr_0.000244140625-ridge_0-run-0-220414-144951/events.out.tfevents.1649940591.nuc-VivoBook-ASUSLaptop-X705FD-N705FD'

log_dirs = list( os.listdir('tf_log') )
metadata_losses = dict()
for log_dir in log_dirs:
    log_metadata = metadataFromLogDirName(log_dir)
    metadata_str = metadataToString(log_metadata)
    if metadata_str not in metadata_losses.keys():
        metadata_losses[ metadata_str ] = list()
    ptf = os.path.join('tf_log', log_dir)
    print(ptf)
    l_files = list(os.listdir(ptf))
    if len(l_files) > 0:
        filename = list(os.listdir(ptf))[0]
        ptf = os.path.join(ptf, filename)
        tau_loss = list()
        for summary in summary_iterator(ptf):
            stuff = summary.summary.value
            if len(stuff) > 0:
                tau_loss.append(stuff[0].simple_value)
        metadata_losses[metadata_str].append( tau_loss )

plt.plot(tau_loss)
plt.show()

expected_len = 30001

#i would then like to plot everything
for key in metadata_losses.keys():
    print(key)
    losses_lists = metadata_losses[key]
    losses_lists = [l for l in losses_lists if len(l) == expected_len]
    print( len(losses_lists) )
    if len(losses_lists) > 0:
        losses_len = len(losses_lists[0])
        losses = np.array(losses_lists)
        mean_line = losses.mean(axis = 0)
        std_line = losses.std(axis = 0)
        up_line = mean_line + 3 * std_line
        low_line = mean_line - 3 * std_line
        df = pd.DataFrame({
            'mean_line' : mean_line,
            'std_line' : std_line,
            'up_line' : up_line,
            'low_line' : low_line,
        })
        df.to_csv('plots/' + key + 'plot_stats.csv', index = False)
