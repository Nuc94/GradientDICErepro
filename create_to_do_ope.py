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
    'Reacher-v2'
]
lrs = [0.01, 0.005, 0.001]
ridges = [0.1, 1]
ridges_zero = [0.1]
discounts = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]

for algo in algos:
    for game in de_games:
        for lr in lrs:
            for discount in discounts:
                ridges_list = ridges_zero
                if game == 'DualDICE':
                    ridges_list = ridges
                for ridge in ridges_list:
                    metadata = metadata_handle.metadataFromVars(game=game, algo=algo, discount=discount, lr=lr, ridge=ridge)
                    entry = dict()
                    entry['runs_required'] = runs_required
                    entry['runs_done'] = 0
                    entry['metadata'] = metadata
                    to_do.append(entry)

with open('to_do_ope_new.json', 'w') as outfile:
    json.dump(to_do, outfile)