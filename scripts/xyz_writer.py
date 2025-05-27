#!/usr/bin/env python3
"""
Convert SDF files to XYZ format compatible with NequIP input requirements.

The output XYZ files contain atomic positions without energy information, formatted as:
<number of atoms>
Properties=species:S:1:pos:R:3 pbc="F F F"
<element> <X> <Y> <Z>

Usage:
    python sdf_to_xyz.py input.sdf -o output_dir [-s {tg,sl}]

Options:
    input.sdf      Path to input SDF file
    -o OUTPUT_DIR  Output directory for XYZ files (required)
    -s {tg,sl}     Save mode: 'tg' for single file or 'sl' for split files [default: tg]
"""

import argparse
from pathlib import Path
import sys
from typing import List, Optional

import numpy as np
from ase import Atoms
from ase.io import write
from rdkit import Chem
from rdkit.Chem.rdchem import Mol


def get_xyz_coordinates(mol: Mol) -> np.ndarray:
    """Extract XYZ coordinates from RDKit molecule object.
    
    Args:
        mol: RDKit molecule object
        
    Returns:
        Numpy array of shape (n_atoms, 3) containing atomic positions
    """
    conf = mol.GetConformer()
    return np.array([
        [conf.GetAtomPosition(i).x, 
         conf.GetAtomPosition(i).y, 
         conf.GetAtomPosition(i).z]
        for i in range(mol.GetNumAtoms())
    ])


def get_atomic_symbols(mol: Mol) -> List[str]:
    """Get list of atomic symbols from RDKit molecule.
    
    Args:
        mol: RDKit molecule object
        
    Returns:
        List of element symbols (e.g., ['C', 'H', 'O'])
    """
    return [atom.GetSymbol() for atom in mol.GetAtoms()]


def validate_output_dir(path: Path) -> Path:
    """Ensure output directory exists and is writable."""
    path.mkdir(parents=True, exist_ok=True)
    if not path.is_dir():
        raise NotADirectoryError(f"Output path {path} is not a directory")
    return path


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "sdf_file",
        type=Path,
        help="Path to input SDF file"
    )
    parser.add_argument(
        "-o", "--output_dir",
        type=Path,
        required=True,
        help="Output directory for XYZ files"
    )
    parser.add_argument(
        "-s", "--save_mode",
        choices=["tg", "sl"],
        default="tg",
        help="Save mode: 'tg' for single file, 'sl' for split files"
    )
    
    args = parser.parse_args()

    try:
        output_dir = validate_output_dir(args.output_dir)
    except (OSError, NotADirectoryError) as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)

    if not args.sdf_file.exists():
        print(f"Error: Input file {args.sdf_file} not found", file=sys.stderr)
        sys.exit(1)

    print(f"Processing {args.sdf_file}")

    try:
        with Chem.SDMolSupplier(str(args.sdf_file), removeHs=False) as supplier:
            for i, mol in enumerate(supplier):
                if mol is None:
                    print(f"Warning: Failed to parse molecule {i} in SDF file", file=sys.stderr)
                    continue

                coordinates = get_xyz_coordinates(mol)
                symbols = get_atomic_symbols(mol)
                
                if args.save_mode == "sl":
                    output_path = output_dir / f"{args.sdf_file.stem}_{i:04d}.xyz"
                    write(str(output_path), Atoms(symbols=symbols, positions=coordinates), format="extxyz")
                else:
                    output_path = output_dir / f"{args.sdf_file.stem}.xyz"
                    write(str(output_path), Atoms(symbols=symbols, positions=coordinates), 
                          format="extxyz", append=(i > 0))

        print(f"Conversion complete. Files saved in {output_dir}")

    except Exception as e:
        print(f"Error during conversion: {str(e)}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
    