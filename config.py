#%%
import json
from pathlib import Path

with open(Path(__file__) / Path('..') / Path('config.json')) as f:
    Config = json.load(f)
print('---------------------------------------------\r\n',
    'Config dictionary:\r\n {}'.format(Config),
    '\r\n---------------------------------------------\r\n')