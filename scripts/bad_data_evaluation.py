#!/usr/bin/env python
"""
Predict and identify bad data indices based on model predictions.

Bad data indices are saved in NPZ files for further analysis. The criteria for bad data is:
|E_pred - E_ref|/sqrt(N) > 0.04 kcal/mol where N is the number of atoms.

Usage:
    conda activate equip_ampere
    python bad_data_predictor.py -id /data/indices/ -ri record.npz -o /output/ -m model.pth -xd /data/xyz/
    or
    python bad_data_predictor.py -id /data/indices/ -ri record.npz -o /output/ -m model.pth -xyz data.xyz
"""

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from ase import io
from ase.units import eV, mol, kcal
from nequip.ase.nequip_calculator import NequIPCalculator


def load_indices(indices_dir: Path, record_file: str) -> Tuple[Dict, Dict]:
    """Load total indices and training indices from files.
    
    Args:
        indices_dir: Directory containing index files
        record_file: Name of the record indices file
        
    Returns:
        Tuple of (total_indices_dict, training_indices_dict)
    """
    total_indices_path = indices_dir / "pair_file_index.json"
    with open(total_indices_path) as f:
        total_indices = json.load(f)
    
    train_indices_path = indices_dir / record_file
    training_indices = np.load(train_indices_path)
    
    return total_indices, training_indices


def get_xyz_files(xyz_dir: str = None, xyz_file: str = None) -> List[Path]:
    """Get list of XYZ files to process from either directory or single file.
    
    Args:
        xyz_dir: Directory containing XYZ files
        xyz_file: Specific XYZ file path (supports glob patterns)
        
    Returns:
        List of Path objects for XYZ files
        
    Raises:
        ValueError: If neither xyz_dir nor xyz_file is provided
    """
    if xyz_dir:
        return sorted(Path(xyz_dir).glob("*.xyz"))
    elif xyz_file:
        path = Path(xyz_file)
        return sorted(path.parent.glob(path.name))
    else:
        raise ValueError("No XYZ files specified - must provide either -xd or -xyz")


def is_bad_data(mol, true_value: float, pred_value: float) -> Tuple[float, int]:
    """Determine if a data point is bad based on energy prediction error.
    Calculate bad data criteria: | 𝐸pred − 𝐸ref|/√𝑁 > 0.04 𝑘𝑐𝑎𝑙/𝑚𝑜𝑙
    
    Args:
        mol: ASE molecule object
        true_value: Reference energy value (kcal/mol)
        pred_value: Predicted energy value (kcal/mol)
        
    Returns:
        Tuple of (error_value, bad_flag) where bad_flag is 1 if bad, 0 otherwise
    """
    atom_count = mol.get_global_number_of_atoms()
    error = abs(pred_value - true_value) / math.sqrt(atom_count)
    return (error, 1) if error > 0.04 else (error, 0)


def process_xyz_file(
    xyz_path: Path,
    calculator: NequIPCalculator,
    total_indices: Dict,
    training_indices: Dict,
    output_dir: Path = None
) -> None:
    """Process a single XYZ file to identify bad data points.
    
    Args:
        xyz_path: Path to XYZ file
        calculator: NequIP calculator for energy predictions
        total_indices: Dictionary of total indices
        training_indices: Dictionary of training indices
        output_dir: Directory to save results (defaults to model directory)
    """
    filename = xyz_path.name
    file_key = filename
    
    # Get training indices for this file (if any)
    train_indices = training_indices.get(file_key, np.array([]))
    train_indices = (train_indices - total_indices[file_key][0]).tolist()
    
    pred_data = {
        'bad_id': [],
        'pred_id': [],
        'error': [],
        'raw_energy_kcal': [],
        'pred_energy_kcal': []
    }
    
    for idx, molecule in enumerate(io.iread(xyz_path)):
        if idx not in train_indices:
            raw_energy = molecule.get_potential_energy()  # Already in kcal/mol
            pred_energy = calculator.get_potential_energy(molecule) * eV/(kcal/mol)
            
            error, is_bad = is_bad_data(molecule, raw_energy, pred_energy)
            
            pred_data['pred_id'].append(idx)
            pred_data['raw_energy_kcal'].append(raw_energy)
            pred_data['pred_energy_kcal'].append(pred_energy)
            pred_data['error'].append(error)
            
            if is_bad:
                pred_data['bad_id'].append(idx)
    
    # Determine output path
    if output_dir:
        output_path = output_dir / f'pred_{xyz_path.stem}'
    else:
        output_path = Path(calculator.model_path).parent / f'pred_{xyz_path.stem}'
    
    np.savez_compressed(f"{output_path}_badid_rawe_prede.npz", **pred_data)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                   formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-id", "--indexdir", 
                       help="Directory containing indices data files",
                       required=True)
    parser.add_argument("-ri", "--recordind",
                       help="File name of record indices file",
                       default='record_indices_new.npz')
    parser.add_argument("-o", "--output",
                       help="Output directory for results")
    parser.add_argument("-m", "--model",
                       help="Path to deployed model (.pth file)",
                       required=True)
    parser.add_argument("-xd", "--xyzdir",
                       help="Directory containing XYZ files")
    parser.add_argument("-xyz", "--xyzfile",
                       help="Specific XYZ file path (supports glob patterns)")
    args = parser.parse_args()
    
    # Load indices and setup calculator
    total_indices, train_indices = load_indices(Path(args.indexdir), args.recordind)
    
    calculator = NequIPCalculator.from_deployed_model(
        model_path=args.model,
        species_to_type_name={"C": "C", "H": "H", "N": "N", "O": "O", "S": "S"},
        device='cpu',
        energy_units_to_eV=(kcal/mol)/eV
    )
    
    # Get XYZ files to process
    try:
        xyz_files = get_xyz_files(args.xyzdir, args.xyzfile)
    except ValueError as e:
        print(f"Error: {e}")
        return
    
    # Process each XYZ file
    output_dir = Path(args.output) if args.output else None
    for xyz_file in xyz_files:
        process_xyz_file(
            xyz_file,
            calculator,
            total_indices,
            train_indices,
            output_dir
        )


if __name__ == "__main__":
    main()