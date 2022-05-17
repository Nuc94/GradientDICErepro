'''
okay, here i plan to write some code to do stuff
'''

import json
import metadata_handle

to_do = list()

runs_required = 10

algos = [
    'GradientDICE',
    'GenDICE',
    'DualDICE'
]
de_games = [
    'BoyansChainTabular-v0',
    'BoyansChainLinear-v0'
]
lrs = [4 ** -1, 4 ** -2, 4 ** -3,4 ** -4, 4 ** -5, 4 ** -6]
ridges = [0, 0.1, 0.001, 0.0001]
ridges_zero = [0]
discounts = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]

for algo in algos:
    for game in de_games:
        for lr in lrs:
            for discount in discounts:
                ridges_list = ridges_zero
                if discount == 1.0:
                    ridges_list = ridges
                for ridge in ridges_list:
                    metadata = metadata_handle.metadataFromVars(game=game, algo=algo, discount=discount, lr=lr, ridge=ridge)
                    entry = dict()
                    entry['runs_required'] = runs_required
                    entry['runs_done'] = 0
                    entry['metadata'] = metadata
                    to_do.append(entry)

with open('to_do_new.json', 'w') as outfile:
    json.dump(to_do, outfile)