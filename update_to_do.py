'''
this script shall take what contained inside the to_do.json and
update according to the actual folders in tf_log 
'''

import os
import json
import metadata_handle

dirs = os.listdir('tf_log')
print(dirs)
print('\n' * 100)

#this one shall be a metadata holder liniojng the string to the actual dict
metadata_metadatas = dict()
#this shall be a dict containing the counter for executions for every
#string in metadata
metadata_counts_de = dict()
metadata_counts_ope = dict()
for dir in dirs:
    dir_meta = metadata_handle.metadataFromLogDirName(dir)
    dir_str = metadata_handle.metadataToString(dir_meta)
    if dir_meta['game'] in metadata_handle.de_games:
        if dir_str not in metadata_counts_de.keys():
            metadata_counts_de[dir_str] = 0
        metadata_counts_de[dir_str] += 1
    elif dir_meta['game'] in metadata_handle.ope_games:
        if dir_str not in metadata_counts_ope.keys():
            metadata_counts_ope[dir_str] = 0
        metadata_counts_ope[dir_str] += 1
    metadata_metadatas[dir_str] = dir_meta

for dir_str in metadata_counts_de.keys():
    print(dir_str)

print(len(metadata_counts_de))

#i shall now update the to_do script
with open('to_do.json', 'r') as infile:
    to_do = json.load(infile)

for task in to_do:
    #i shall seek the metadata
    mt = task['metadata']
    #before doing everything i just want to convert all terms in float from to_do
    mt['discount'] = float(mt['discount'])
    mt['lr'] = float(mt['lr'])
    mt['ridge'] = float(mt['ridge'])
    mt_str = metadata_handle.metadataToString(mt)
    task['runs_done'] = metadata_counts_de[mt_str]

with open('to_do_upd.json', 'w') as outfile:
    json.dump(to_do, outfile)

#i would need the same thing for ope
with open('to_do_ope.json', 'r') as infile:
    to_do = json.load(infile)

for task in to_do:
    #i shall seek the metadata
    mt = task['metadata']
    #before doing everything i just want to convert all terms in float from to_do
    mt['discount'] = float(mt['discount'])
    mt['lr'] = float(mt['lr'])
    mt['ridge'] = float(mt['ridge'])
    mt_str = metadata_handle.metadataToString(mt)
    if mt_str in metadata_counts_ope.keys():
        task['runs_done'] = metadata_counts_ope[mt_str]

with open('to_do_ope_upd.json', 'w') as outfile:
    json.dump(to_do, outfile)