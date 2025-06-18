#!/usr/bin/env python
"""
Select frames from extxyz files according to indices and save them to new extxyz file.

Indices are randomly selected from indices file and will be saved to npz file.
Data are selected according to average value of bad data dE from three models.

Usage:
    python xxx.py -id indices/json/files/dir -pred pred/files/dir/prefix 
                  -xd total/xyz/files/dir -o fg --generate_new
"""

import argparse
import bisect
import json
from pathlib import Path
from random import shuffle
from typing import Dict, List, Tuple

import numpy as np
from ase.io import read, write


def load_indices(indices_path: Path) -> Tuple[Dict, int]:
    """Load indices from JSON files."""
    if not indices_path:
        raise FileNotFoundError("No indices JSON files found in the specified directory")
    
    with open(indices_path, 'r') as f:
        indices_data = json.load(f)
    
    total_frames = indices_data.pop('total')
    return indices_data, total_frames


def load_prediction_errors(indices_path: Path, pred_prefix: str, total_indices_d: Dict, total_frames: int, model_versions: int = 3) -> Tuple[Dict, List]:
    """Load prediction errors from multiple model versions.
    model_versions = 3, number of parallel trained models (currently training 3 models in parallel)
    """
    pred_error_dict = {}
    bad_rates = []
    
    for i in range(model_versions):
        version = f'v{i + 1}'
        pred_dir = indices_path / version / f'{pred_prefix}_{i + 1}'
        bad_data_files = list(pred_dir.glob('pred_*_badid_rawe_prede.npz'))
        
        bad_id_dict = {}
        pred_id_dict = {}
        pred_error_dict = {}
        num_bad = 0
        
        for bad_file in bad_data_files:
            parts = bad_file.name.split('_')
            if len(parts) == 6:
                pairname = f'{parts[1]}_{parts[2]}'
            elif len(parts) == 7:
                pairname = f'{parts[1]}_{parts[2]}_{parts[3]}'
                
            min_index = total_indices_d[f'{pairname}.xyz'][0]
            file_data = np.load(bad_file)
            
            bad_id_array = file_data['bad_id'] + min_index
            bad_id_dict[pairname] = bad_id_array
            num_bad += len(bad_id_array)
            
            pred_id_array = file_data['pred_id'] + min_index
            pred_id_dict[pairname] = pred_id_array
            pred_error_dict[pairname] = file_data['error']
        
        print(f'For model {version}, {num_bad} bad data detected ({num_bad / total_frames * 100:.2f}%)')
        bad_rates.append(num_bad / total_frames)
        pred_error_dict[version] = [
            np.concatenate(list(pred_id_dict.values())),
            np.concatenate(list(pred_error_dict.values()))
        ]
    
    return pred_error_dict, bad_rates


def calculate_mean_errors(pred_error_dict: Dict, model_versions: int) -> Tuple[np.ndarray, int]:
    """Calculate mean errors across model versions."""
    sum_errors = np.zeros_like(pred_error_dict['v1'][1])
    for version in pred_error_dict:
        sum_errors += pred_error_dict[version][1]
    
    mean_errors = sum_errors / model_versions
    bad_indices = np.where(mean_errors > 0.04)[0]
    bad_ids = pred_error_dict['v1'][0][bad_indices]
    
    return bad_ids, len(bad_indices)


def which_file(index: int, indices_rlimit: List[int], filename_rlimit: List[str]) -> str:
    """Match index to filename using binary search."""
    file_idx = bisect.bisect_left(indices_rlimit, index)
    return filename_rlimit[file_idx]


def filter_wrong_molecules(frames: List) -> Tuple[List, List]:
    """Filter out molecules with unrealistic energies."""
    del_indices = []
    for i, mol in enumerate(frames):
        energy = mol.get_potential_energy()
        if energy < -200 or energy > 150:
            del_indices.append(i)
    
    return [mol for i, mol in enumerate(frames) if i not in del_indices], del_indices


def filter_wrong_indices(del_indices: List[int], old_indices: List[int]) -> List[int]:
    """Remove indices corresponding to wrong molecules."""
    old_array = np.array(old_indices)
    mask = np.ones(len(old_indices), dtype=bool)
    mask[del_indices] = False
    return old_array[mask].tolist()


def save_with_backup(filepath: Path, data: Dict, backup_suffix: str = '_prev') -> None:
    """Save data with backup of existing file."""
    if filepath.exists():
        backup_path = filepath.with_name(f"{filepath.stem}{backup_suffix}{filepath.suffix}")
        filepath.rename(backup_path)
    
    if filepath.suffix == '.npz':
        np.savez_compressed(str(filepath), **data)
    else:
        raise ValueError("Only .npz files are supported for backup saving")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-id", "--indp", help="Path to indices JSON files", required=True)
    parser.add_argument("-pred", "--predir", help="Prefix of prediction file directory, e.g. al_iter_0", required=True)
    parser.add_argument("-xd", "--xyzdir", help="Directory path to XYZ files", required=True)
    parser.add_argument("-o", "--output", help="Output prefix for train set files", default='iter_new')
    parser.add_argument("--generate_new", help="Generate new training set", action='store_true')
    args = parser.parse_args()

    # Load index data
    total_indices_d, total_frames = load_indices(Path(args.indp))
    
    # Load prediction errors
    pred_error_dict, bad_rates = load_prediction_errors(Path(args.indp), args.predir, total_indices_d, total_frames)
    if not pred_error_dict:
        print('No prediction error data found')
        return

    # Calculate mean errors and select bad frames
    mean_bad_ids, mean_bad_num = calculate_mean_errors(pred_error_dict, len(bad_rates))
    print(f'Average: {mean_bad_num} bad data detected ({mean_bad_num / total_frames * 100:.2f}%)')

    # Determine number of frames to select
    percent_select = 5
    if mean_bad_num / total_frames > 0.05:
        num_to_select = int(mean_bad_num * percent_select / 100)
        print(f'Selecting {num_to_select} bad data points')
    else:
        print('Bad data percentage is lower than 5%, selecting all bad data')
        num_to_select = mean_bad_num

    selected_indices = np.random.choice(mean_bad_ids, size=num_to_select, replace=False)
    selected_indices = np.sort(selected_indices)

    # Organize selected indices by file
    indices_rlimit = sorted(total_indices_d.values(), key=lambda x: x[1])
    filenames_rlimit = [k for k, _ in sorted(total_indices_d.items(), key=lambda x: x[1][1])]
    
    selected_data = {}
    for idx in selected_indices:
        filename = which_file(idx, [x[1] for x in indices_rlimit], filenames_rlimit)
        selected_data.setdefault(filename, []).append(idx) 

    # Process and save selected frames
    output_path = Path(args.indp)
    train_file = output_path / f"{args.output}_train.xyz"
    
    if train_file.exists():
        train_file.unlink()

    for filename, indices in selected_data.items():
        molecules = read(str(Path(args.xyzdir) / filename), index=':')
        local_indices = np.array(indices) - total_indices_d[filename][0]
        frames = [molecules[i] for i in local_indices]
        
        filtered_frames, del_indices = filter_wrong_molecules(frames) ??
        write(str(train_file), filtered_frames, append=True)
        
        if del_indices:
            print(f'Removed {len(del_indices)} invalid frames from {filename}')
            selected_data[filename] = filter_wrong_indices(del_indices, indices)

    # Shuffle and save final training set
    molecules = read(str(train_file), index=':')
    shuffle(molecules)
    write(str(train_file), molecules)

    # Save indices
    record_file = output_path / f"{args.output}_record_indices_new.npz"
    if record_file.exists():
        record_data = dict(np.load(str(record_file)))
    else:
        record_data = {k: np.array([]) for k in total_indices_d}
    
    for filename in total_indices_d:
        record_data[filename] = np.concatenate([
            record_data[filename],
            np.array(selected_data.get(filename, []))
        ])
    
    save_with_backup(record_file, record_data)
    
    # Save selected data
    output_file = output_path / f"{args.output}.npz"
    save_with_backup(output_file, selected_data)


if __name__ == "__main__":
    main()
