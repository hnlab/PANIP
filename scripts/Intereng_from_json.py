#! /usr/bin/env python
"""
Analysis of QM energy JSON files:

- Function:
  * Parse QM energy JSON file
  * Output pairwise interaction energies into a new JSON file

- Formula:
  * E_interaction = E_AB - E(A) - E(B)

- Units:
  * ORCA output energies are in Hartree (Eh)
  * Conversion factors:
      1 Eh = 27.2113834 eV
      1 eV = 23.0605 kcal/mol
"""

import json
from os import name
from pathlib import Path
import argparse
import gzip

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("jsonfile", help="path of json file")
parser.add_argument("-o", "--output", help="path of output json file")
args = parser.parse_args()

inp = Path(args.jsonfile)
oup = Path(args.output) if args.output else Path(f"inteng_{inp.stem}")

print(f"Processing {inp.name}")
if inp.suffix == ".gz" and inp.name.endswith(".json.gz"):
    with gzip.open(inp, "rt", encoding="utf-8") as f:
        qm_data = json.load(f)
else:
    with inp.open("r", encoding="utf-8") as f:
        qm_data = json.load(f)

pair_dict = {}
fraga_dict = {}
fragb_dict = {}
inter_energy_dict = {}
flag = ""
flag_num = -1
for i in qm_data:
    name = i["name"]
    if "energy" in i:
        energy = i["energy"]
    # for several unusual situation
    elif "single point energy" not in i and "energy" not in i:
        energy = 99999999
        print(f'lack energy !!!{flag_num}, pairname: {name}')
    if not name.endswith("fa") and not name.endswith("fb"):
        flag_num += 1
        pair_dict[flag_num] = energy
    elif name.endswith("fa"):
        fraga_dict[flag_num] = energy
    elif name.endswith("fb"):
        fragb_dict[flag_num] = energy

for p, e in pair_dict.items():
    fa_energy = fraga_dict[p]
    fb_energy = fragb_dict[p]
    inter_energy = e - fa_energy - fb_energy
    if e == 99999999 or fa_energy == 99999999 or fb_energy == 99999999:
        print(f'the lack index in inteng.json is {p}, energy is {e}')
    inter_energy_dict[p] = inter_energy * 27.2113834 * 23.0605

with open(str(oup), "w") as oupf:
    json.dump(inter_energy_dict, oupf, indent=2)

print(f"data saved in {oup}")