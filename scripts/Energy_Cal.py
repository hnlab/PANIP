#!/bin/env python
"""
Energy calculation workflow:

- Methods:
  * r²SCAN-3c
  * ωB97X-D3/def2-TZVP (via ORCA)

- BSSE correction:
  * Applied for all systems except ACET

- Requirements:
  * RDKit
  * fullspace (group-developed package)

Install fullspace:
    git clone git@github.com:hnlab/fullspace.git
    cd fullspace
    python3 setup.py install --user

Test run:
    python Energy_Cal.py ~/PANIP/examples/ACEM_MIND.sdf -o qm_r2scan -qm r2scan

Calculate interaction energy:
    python Intereng_from_json.py qm_r2scan.json
"""

from rdkit import Chem
from fullspace import fragment, tools
import argparse

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("pairs", help="pairs in sdf format.")
# parser.add_argument(setting, help="specify DFT, Basis set and other setting")
parser.add_argument(
    "-o", "--output", default="qm", help="default output QM data in qm.json.gz."
)
parser.add_argument(
    "-qm", "--qm_method", default="r2scan", help="specific qm method r2scan (default) or wb97x"
)
args = parser.parse_args()

# calulate pair total energy
chgfrags = {'ACET':-1, 'ETAM':1, 'MIMM':1, 'MGDM':1}
mols = []
with Chem.SDMolSupplier(args.pairs, removeHs=False) as suppl:
    for mol in suppl:
        species = [a.GetSymbol() for a in mol.GetAtoms()]
        new_gto = [" " for i in mol.GetAtoms()]
        frag_name = mol.GetProp("FRAG_NAME")
        frag_atom_num = mol.GetProp("FRAG_ATOM_NUM").strip().split()[1::2]
        charge = 0
        frag_namelist = frag_name.strip().split()
        for n in frag_namelist:
            if n in chgfrags:
                charge += chgfrags[n]
        if args.qm_method == "r2scan":
            diff_fuc = 'newgto "ma-def2-TZVP" end'
        elif args.qm_method == "wb97x":
            diff_fuc = 'newgto "ma-def2-TZVPP" end'
        if "ACET   " in frag_name:
            new_gto[5] = diff_fuc
            new_gto[6] = diff_fuc
        if "   ACET" in frag_name:
            new_gto[-1] = diff_fuc
            new_gto[-2] = diff_fuc

        BSSE = False if args.qm_method == "r2scan" or "ACET" in frag_name else True
        
        traj = [
            mol.GetConformer().GetAtomPosition(a) for a in range(len(mol.GetAtoms()))
        ]
        coord = [[pos.x, pos.y, pos.z] for pos in traj]
        x1 = traj[0].x  # x of the first atom
        x2 = traj[-1].x  # x of the last atom
        name = f"{str(mol.GetProp('PDB_ID'))} {frag_name} {x1} {x2}"
        m = fragment.Mol(
            name,
            species=species,
            refCoord=coord,
            charge=charge,
            new_gto=new_gto,
            reset_com=False,
        )
        mols.append(m)

        # for fragment
        fa_num = int(frag_atom_num[0])
        name_fa = f'{name} fa'
        name_fb = f'{name} fb'
        charge_fa = chgfrags[frag_namelist[0]] if frag_namelist[0] in chgfrags else 0
        charge_fb = chgfrags[frag_namelist[1]] if frag_namelist[1] in chgfrags else 0
        if BSSE is False:
            species_fa = species[:fa_num]
            species_fb = species[fa_num:]
            new_gto_fa = new_gto[:fa_num]
            new_gto_fb = new_gto[fa_num:]
            coord_fa = coord[:fa_num]
            coord_fb = coord[fa_num:]
        elif BSSE is True:
            # add ghost atom label
            species_fa = species[:fa_num] + [f'{ele} :' for ele in species[fa_num:]]
            species_fb = [f'{ele} :' for ele in species[:fa_num]] + species[fa_num:]
            new_gto_fa = new_gto
            new_gto_fb = new_gto
            coord_fa = coord
            coord_fb = coord


        fa = fragment.Mol(
            name_fa,
            species=species_fa,
            refCoord=coord_fa,
            charge=charge_fa,
            new_gto=new_gto_fa,
            reset_com=False,
        )
        mols.append(fa)
        fb = fragment.Mol(
            name_fb,
            species=species_fb,
            refCoord=coord_fb,
            charge=charge_fb,
            new_gto=new_gto_fb,
            reset_com=False,
        )
        mols.append(fb)

# if BSSE is calculated, Pmodel will be added to fragment setting by tools.orca_runall() automatly\
if args.qm_method == "r2scan":
    data = tools.orca_runall(
        mols=mols,
        parse=True,
        setting="! r2SCAN-3c\n",
    )
elif args.qm_method == "wb97x":
    data = tools.orca_runall(
        mols=mols,
        parse=True,
        # setting="%pal nprocs  8 end\n! engrad wB97X-D3BJ  def2-TZVPP def2/J def2-TZVPP/C RIJCOSX\n"
        setting="! engrad wB97X-D3BJ  def2-TZVPP def2/J def2-TZVPP/C RIJCOSX\n"
    )

tools.json_gzip_dump(data,f'{args.output}.json.gz' )
