#!/usr/bin/env python
"""
Predict molecular energy from XYZ files using an ensemble of wB98-ML models.  

Note:
    Please refer to the corresponding model's README.md for applicable dimers.

Usage:
    python predict_energy_parallel.py -xyz <xyz_path> -md <models_dir> -m <models_name> -od <output_dir> [--mlp]

Options:
    -xyz, --xyzfile      Path to input XYZ file (required)
    -md, --models_dir    Directory containing trained models [default: ./models]
    -od, --output        Output directory for predictions [default: ./predictions]
    --mlp                Enable multiprocessing (recommended for faster inference)
    -w, --workers        Number of worker processes to use (only with --mlp) [default: 2]

Author: Lejia Zeng
Version: 1.0.0
"""

# Core imports
from pathlib import Path
import argparse
import time
import numpy as np
from typing import List, Tuple, Dict

# Third-party imports
from ase import io
from ase.units import eV, mol, kcal
from nequip.ase.nequip_calculator import NequIPCalculator

# Constants
SPECIES_MAP = {"C": "C", "H": "H", "N": "N", "O": "O", "S": "S"}
ENERGY_CONVERSION = (kcal/mol)/eV

# --------------------------
#  Core Functionality
# --------------------------

def load_models(models_dir: Path, model_name: str) -> List[Path]:
    """Load deployed model paths based on functional group naming convention."""
    def model_subdir(name: str) -> Path:
        return models_dir / ("MIMX" if "MIM" in name else name.upper())

    dir_m = model_subdir(model_name)
    # Validate model directories
    if dir_m.exists() and dir_m.is_dir():
        pass
    else:
        raise FileNotFoundError(f"No valid model directories found for {model_name}")

    return sorted(dir_m.glob("*.pth"))

def predict_energy(calc: NequIPCalculator, frames) -> float:
    """Calculate potential energy with unit conversion."""
    return calc.get_potential_energy(frames) * eV/(kcal/mol)
# --------------------------
#  Pipeline Orchestration
# --------------------------

def create_calculator(model_path: str) -> NequIPCalculator:
    """Initialize NequIP calculator with standardized settings."""
    return NequIPCalculator.from_deployed_model(
        model_path=model_path,
        species_to_type_name=SPECIES_MAP,
        device='cpu',
        energy_units_to_eV=ENERGY_CONVERSION
    )

def process_single_model(xyz_path: Path, model_path: str) -> List[float]:
    """Process all molecules in XYZ file with one model."""
    calc = create_calculator(model_path)
    energy_list = [predict_energy(calc, frames) for frames in io.iread(xyz_path)]
    if 'MIND' in model_path:
        energy_list = [e / (27.2113834 * 23.0605) for e in energy_list]
    return energy_list

# --------------------------
#  Main Execution
# --------------------------

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-xyz", "--xyzfile", required=True, type=Path)
    parser.add_argument("-md", "--models_dir", type=Path, default=Path("../models"))
    parser.add_argument("-m", "--model_name", type=Path, default=Path("GLOBAL"))
    parser.add_argument("-od", "--output", type=Path, default=Path("../predictions"))
    parser.add_argument("--mlp", action="store_true")
    parser.add_argument("-w", "--workers", type=int, default=2, help="the number of worker processes to use")
    args = parser.parse_args()

    # Prepare output directory
    args.output.mkdir(parents=True, exist_ok=True)

    # fg_a, fg_b = Path(args.xyzfile).stem.split("_")[:2]
    
    try:
        # Load models
        models = load_models(args.models_dir, args.model_name)
        print(f"Loaded {len(models)} models for {args.model_name}")

        # Run predictions
        start_time = time.time()
        if args.mlp:
            from multiprocessing import Pool
            
            with Pool(args.w) as pool:
                results = pool.starmap(process_single_model, 
                                    [(args.xyzfile, str(m)) for m in models])
        else:
            results = [process_single_model(args.xyzfile, str(m)) for m in models]

        # Average predictions
        avg_energy = np.mean(results, axis=0)
        output_path = args.output / f"{args.xyzfile.stem}_predictions.npy"
        np.save(output_path, avg_energy)

        print(f"Prediction completed in {time.time()-start_time:.2f}s")
        print(f"Results saved to {output_path}")

    except Exception as e:
        print(f"Error: {str(e)}")
        exit(1)

if __name__ == "__main__":
    main()
