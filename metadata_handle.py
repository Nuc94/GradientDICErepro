'''
this shall be a little modult to extract and handle stuff from folder names to get type of game and so on
'''

import json

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
    
    if metadata['game'] != 'unkown':
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
        metadata['discount'] = float(discount)
        metadata['lr'] = float(lr)
        metadata['ridge'] = float(ridge)
    return metadata

def metadataFromVars(game, algo, discount, lr, ridge):
    metadata = {
        'game' : game,
        'algo' : algo,
        'discount' : float(discount),
        'lr' : float(lr),
        'ridge' : float(ridge)
    }
    return metadata

def metadataToString(metadata):
    return json.dumps(metadata).replace('\\', '')