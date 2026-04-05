#!/usr/bin/env python
"""generate index for pairs saved under one directory, and save index in one json file
{'fga_fgb_num.xyz':[min_index, max_index],...,'total':total_num}
usage:
python set_pair_index.py xyz/dir/path output/dir/path
"""
import json
from ase.io import read
from pathlib import Path
import sys

file_path = Path(sys.argv[1])
xyz_fs = sorted(file_path.glob('*.xyz'))
oup_f = Path(sys.argv[2]) / 'pair_file_index.json'

index_d = {}
index_head = 0
index_tail = -1
for file in xyz_fs:
    f_name = file.name
    index_head = index_tail + 1
    atoms = read(file, format='extxyz',index=':')
    num_frames = len(atoms)
    index_tail += num_frames
    index_d[f_name] = [index_head, index_tail]

index_d['total'] = index_tail + 1

with open(oup_f, 'w') as f:
    json.dump(index_d, f, indent=4)
    print(f'Index file saved to {oup_f}')