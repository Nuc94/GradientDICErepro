'''
ok, if i were to need to increase the level of
'''

import json

filename = 'to_do_ope.json'
runs_to_do = 30

with open(filename, 'r') as infile:
    to_do = json.load(infile)

for task in to_do:
    task['runs_required'] = runs_to_do

with open(filename, 'w') as outfile:
    json.dump(to_do, outfile)
