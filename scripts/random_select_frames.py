#!/usr/bin/env python
"""Select frames from extxyz/xyz files according to indices and save them to a new extxyz file.

This script was refactored for importability and clarity while keeping the original
selection logic. It supports two modes:
  - --init: randomly select a percentage of frames from the whole dataset
  - default: use prediction NPZ files (from multiple model versions) to identify
    "bad" frames (mean error > 0.04) and select a percentage of those

The script writes a shuffled training file and updates record/selection NPZ files in the indices directory.
"""

from __future__ import annotations

import argparse
import json
import bisect
from pathlib import Path
from random import shuffle
from typing import Dict, List, Tuple
import re

import numpy as np
from ase.io import read, write


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("-id", "--indir", help="indices json files path", required=True)
    p.add_argument("-pc", "--percent", help="percent of data to select", default=5, type=int)
    p.add_argument("-pred", "--predir", nargs="+", help="prediction file directories for parallel trained models", default=["pred"])
    p.add_argument("-xd", "--xyzdir", help="total xyz files path", required=True)
    p.add_argument("-o", "--output", help="prefix of train_set.xyz and others ", default="iter_new")
    p.add_argument("--init", action="store_true", help="whether to initialize with random selection from whole dataset")
    return p.parse_args()


def load_pair_index(indir: Path) -> Tuple[Dict[str, List[int]], int]:
    path = indir / "pair_file_index.json"
    if not path.exists():
        raise FileNotFoundError(f"pair_file_index.json not found in {indir}")
    with open(path, "r") as fh:
        d = json.load(fh)
    total = d.pop("total")
    return d, total


def which_file(index: int, indices_rlimit_list: List[int], filename_rlimit_list: List[str]) -> str:
    """Match a global index to a filename using right-limits and bisect."""
    file_ind = bisect.bisect_left(indices_rlimit_list, index)
    return filename_rlimit_list[file_ind]

def select_indices_from_predictions(indir: Path, total_indices_d: Dict[str, List[int]], percent_select: int) -> np.ndarray:
    """Collect prediction NPZs from multiple model versions and return selected indices.

    Looks for files matching pattern 'pred_*_badid_rawe_prede.npz' under
    {predir} as in the original script.
    """
    pred_dirs = [Path(i) for i in args.predir]
    pred_error_dict = {}

    for i, pred_dir in enumerate(pred_dirs):
        files = list(pred_dir.glob("pred_*_badid_rawe_prede.npz"))

        predid_d = {}
        prederror_d = {}
        num_bad = 0
        for f in files:
            fname = re.search(r"pred_(.*?)_badid_rawe_prede.npz", f.name).group(1)

            if f"{fname}.xyz" not in total_indices_d:
                continue
            min_ind = total_indices_d[f"{fname}.xyz"][0]
            data = np.load(f)
            bad_id_array = data["bad_id"] + min_ind
            pred_id_array = data["pred_id"] + min_ind
            predid_d[fname] = pred_id_array
            prederror_d[fname] = data["error"]
            num_bad += bad_id_array.size

        if num_bad > 0:
            print(f"Under {pred_dir.stem}, {num_bad} bad data is detected, account for {num_bad / total_frames * 100}%")
            pred_error_dict[f"model_{i+1}"] = [np.concatenate(list(predid_d.values())), np.concatenate(list(prederror_d.values()))]

    if not pred_error_dict:
        raise RuntimeError("No prediction error files found; cannot select indices")

    # calculate mean error across parallel models
    sum_arr = np.zeros_like(pred_error_dict['model_1'][1])
    for ver in pred_error_dict:
        sum_arr += pred_error_dict[ver][1]
    mean_arr = sum_arr / len(pred_dirs)
    mean_bad = np.where(mean_arr > 0.04)[0]
    mean_bad_id = pred_error_dict['model_1'][0][mean_bad]
    mean_bad_num = mean_bad.shape[0]
    print(f'for average, {mean_bad_num} bad data is detected, account for {mean_bad_num / total_frames * 100}%')

    if mean_bad_num / total_frames > 0.05:
        num_to_select = int(mean_bad_num * percent_select / 100)
        print(f"{num_to_select} bad data is selected")
    else:
        print('bad data percent is lower than 5%')
        num_to_select = mean_bad_num

    return np.random.choice(mean_bad_id, size=num_to_select, replace=False)


def save_record_and_selection(indir: Path, output_prefix: str, total_indices_d: Dict[str, List[int]], select_d: Dict[str, List[int]]):
    record_ind_f = indir / f"{output_prefix}_record_indices.npz"
    if record_ind_f.exists():
        record_indices_d = dict(np.load(str(record_ind_f)))
    else:
        record_indices_d = {k: np.asarray([]) for k in total_indices_d}

    record_ind_new_d = {}
    for k in total_indices_d:
        if k not in select_d:
            select_d[k] = []
        record_ind_new_d[k] = np.concatenate((record_indices_d.get(k, np.asarray([])), np.asarray(select_d[k]))).astype(int)

    if record_ind_f.exists():
        record_ind_f_old = indir / f"record_indices_prev.npz"
        record_ind_f.rename(record_ind_f_old)
    np.savez_compressed(str(record_ind_f), **record_ind_new_d)

    # save select_d
    oupath = indir / f"{output_prefix}.npz"
    if oupath.exists():
        select_f_old = indir / f"{output_prefix}_iter_prev.npz"
        oupath.rename(select_f_old)
    np.savez_compressed(str(oupath), **select_d)


def main():
    global args, total_frames
    args = parse_args()
    indir = Path(args.indir)
    total_indices_d, total_frames = load_pair_index(indir)

    percent_select = args.percent

    if args.init:
        print(f'select {percent_select}% from whole dataset as initial training set')
        num_to_select = int(total_frames * percent_select / 100)
        indices_select = np.random.choice(np.arange(total_frames), size=num_to_select, replace=False)
    else:
        indices_select = select_indices_from_predictions(indir, total_indices_d, percent_select)


    indices_select = np.sort(indices_select)

    # build right-limit lists for mapping global index -> filename
    indices_rlimit_list = []
    filename_rlimit_d = {}
    for f in total_indices_d:
        index_rlimit = total_indices_d[f][1]
        indices_rlimit_list.append(index_rlimit)
        filename_rlimit_d[index_rlimit] = f
    indices_rlimit_list.sort()
    filename_rlimit_list = [filename_rlimit_d[i] for i in indices_rlimit_list]

    select_d: Dict[str, List[int]] = {}
    for ind in indices_select:
        match_filen = which_file(ind, indices_rlimit_list, filename_rlimit_list)
        select_d.setdefault(match_filen, []).append(int(ind))

    # get frames and write to train file
    train_file = Path(args.indir) / f"{args.output}_train.xyz"
    if train_file.exists():
        if args.init:
            print(f"Removing existing train file: {train_file}")
            train_file.unlink()
        else:
            train_file.rename(Path(args.indir) / f"iter_prev_train.xyz")

    for xyzfilename, inds in select_d.items():
        mols = read(str(Path(args.xyzdir) / xyzfilename), index=':')
        ind_begin_0 = np.asarray(inds) - total_indices_d[xyzfilename][0]
        frames = [mols[i] for i in ind_begin_0]
        write(str(train_file), frames, append=True)

    # save record & selection npz files
    save_record_and_selection(indir, args.output, total_indices_d, select_d)

    # shuffle final train file
    if train_file.exists():
        unshf_mols = read(str(train_file), index=':')
        shuffle(unshf_mols)
        write(str(train_file), unshf_mols)


if __name__ == '__main__':
    main()