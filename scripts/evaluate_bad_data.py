#!/usr/bin/env python
"""Predict and identify bad data indices based on model predictions.

This script inspects XYZ files, predicts energies using NequIP model,
and saves NPZ summary files containing predicted energies, raw energies, errors
and lists of bad indices (where |E_pred - E_ref|/sqrt(N_atoms) > 0.04 kcal/mol).

The script is refactored for clarity and community distribution: functions are
typed, paths are validated, and outputs are saved with clear names.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from ase import io
from ase.units import eV, mol, kcal
from nequip.ase.nequip_calculator import NequIPCalculator


def load_indices(indices_dir: Path, record_file: str) -> Tuple[Dict[str, List[int]], Dict[str, np.ndarray]]:
    """Load total index mapping and previous record indices (if present).

    Returns:
        total_indices: mapping filename -> [min_index, max_index]
        training_indices: mapping filename -> array of recorded indices
    """
    total_indices_path = indices_dir / "pair_file_index.json"
    if not total_indices_path.exists():
        raise FileNotFoundError(f"pair_file_index.json not found in {indices_dir}")

    with open(total_indices_path, "r") as fh:
        total_indices = json.load(fh)

    training_indices_path = indices_dir / record_file
    if training_indices_path.exists():
        training_indices = dict(np.load(str(training_indices_path)))
    else:
        training_indices = {k: np.asarray([]) for k in total_indices}

    return total_indices, training_indices


def get_xyz_files(xyz_dir: Optional[str] = None, xyz_file: Optional[str] = None) -> List[Path]:
    """Return a sorted list of XYZ file paths to process.

    Provide either `xyz_dir` or `xyz_file` (glob allowed). If both are None,
    raise ValueError.
    """
    if xyz_dir:
        return sorted(Path(xyz_dir).glob("*.xyz"))
    if xyz_file:
        p = Path(xyz_file)
        return sorted(p.parent.glob(p.name))
    raise ValueError("No XYZ files specified — provide either -xd or -xyz")


def is_bad_data(reader_xyz_item, true_value: float, pred_value: float) -> Tuple[float, bool]:
    """Compute normalized error and whether the frame is bad.

    Uses atom count via len(reader_xyz_item) (compatible with ASE Atoms).
    """
    atom_count = len(reader_xyz_item)
    if atom_count <= 0:
        raise ValueError("Molecule has zero atoms")
    error = abs(pred_value - true_value) / math.sqrt(atom_count)
    return error, (error > 0.04)


def process_xyz_file(
    xyz_path: Path,
    calculator: NequIPCalculator,
    total_indices: Dict[str, List[int]],
    training_indices: Dict[str, np.ndarray],
    output_dir: Optional[Path] = None,
) -> None:
    """Process a single XYZ file and save a pred_*_badid_rawe_prede.npz result file.

    The output file contains keys: bad_id, pred_id, error, raw_energy_kcal, pred_energy_kcal
    """
    filename = xyz_path.name

    # training indices for this file (global -> local mapping handled by caller)
    train_idx_global = training_indices.get(filename, np.asarray([]))
    train_idx_local = (train_idx_global - total_indices[filename][0]).tolist() if train_idx_global.size else []

    pred_data = {
        "bad_id": [],
        "pred_id": [],
        "error": [],
        "raw_energy_kcal": [],
        "pred_energy_kcal": [],
    }

    # iterate frames
    for idx, reader_xyz_item in enumerate(io.iread(str(xyz_path))):
        if idx in train_idx_local:
            # skip frames that are already in training set
            continue

        # raw energy is expected to be in kcal/mol in the dataset
        raw_energy = reader_xyz_item.get_potential_energy()

        # convert predicted energy returned by NequIPCalculator to kcal/mol
        pred_energy = calculator.get_potential_energy(reader_xyz_item) * eV / (kcal / mol)

        error, bad_flag = is_bad_data(reader_xyz_item, raw_energy, pred_energy)

        pred_data["pred_id"].append(idx)
        pred_data["raw_energy_kcal"].append(raw_energy)
        pred_data["pred_energy_kcal"].append(pred_energy)
        pred_data["error"].append(error)
        if bad_flag:
            pred_data["bad_id"].append(idx)

    # build output path
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        out_base = output_dir / f"pred_{xyz_path.stem}"
    else:
        # fallback to same dir as model (calculator.model_path may be a file inside model dir)
        model_parent = Path(calculator.model_path).parent
        out_base = model_parent / f"pred_{xyz_path.stem}"

    out_file = f"{out_base}_badid_rawe_prede.npz"
    # convert lists to arrays for compact storage
    np.savez_compressed(out_file,
                        bad_id=np.asarray(pred_data["bad_id"]),
                        pred_id=np.asarray(pred_data["pred_id"]),
                        error=np.asarray(pred_data["error"]),
                        raw_energy_kcal=np.asarray(pred_data["raw_energy_kcal"]),
                        pred_energy_kcal=np.asarray(pred_data["pred_energy_kcal"]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-id", "--indexdir", required=True, help="Directory containing indices data files")
    parser.add_argument("-ri", "--recordind", default="iter_new_record_indices.npz", help="File name of record indices file")
    parser.add_argument("-o", "--output", help="Output directory for results")
    parser.add_argument("-m", "--model", required=True, help="Path to deployed model (.pth file)")
    parser.add_argument("-xd", "--xyzdir", help="Directory containing XYZ files")
    parser.add_argument("-xyz", "--xyzfile", help="Specific XYZ file path (supports glob patterns)")
    args = parser.parse_args()

    indexdir = Path(args.indexdir)
    total_indices, training_indices = load_indices(indexdir, args.recordind)

    # initialize calculator; default to CPU for community-friendly behavior
    calc = NequIPCalculator.from_deployed_model(
        model_path=str(args.model),
        species_to_type_name={"C": "C", "H": "H", "N": "N", "O": "O", "S": "S"},
        device="cpu",
        energy_units_to_eV=(kcal / mol) / eV,
    )

    try:
        xyz_files = get_xyz_files(args.xyzdir, args.xyzfile)
    except ValueError as exc:
        print(f"Error: {exc}")
        return

    out_dir = Path(args.output) if args.output else Path(args.indexdir) / "pred"
    out_dir.mkdir(exist_ok=True)

    for xyz in xyz_files:
        try:
            process_xyz_file(xyz, calc, total_indices, training_indices, out_dir)
        except Exception as e:
            print(f"Failed processing {xyz}: {e}")
    print(f"results saved to {out_dir}" )

if __name__ == "__main__":
    main()