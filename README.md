# wB97X-ML
Robust machine learning interatomic potentials (MLIPs) that achieve accuracy comparable to the ωB97X-D3BJ/def2-TZVPP quantum mechanical method on non-covalent interactions.  

---

## Requirements
wB97X-ML is built on NequIP, please install NequIP first.   
- Python >= 3.9
- [NequIP](https://github.com/mir-group/nequip/releases/tag/v0.5.6) == 0.5.6
<!-- - ASE (Atomic Simulation Environment)  -->

### Quick Setup
```bash
# Create and activate a conda environment (recommended)  
conda create -n nequip-env python=3.10
conda activate nequip-env

# Install PyTorch with CUDA 11.3 (adjust based on your driver)  
conda install pytorch==1.11.0 cudatoolkit=11.3 -c pytorch

# Install Nequip 0.5.6 and dependencies  
wget https://github.com/mir-group/nequip/archive/refs/tags/v0.5.6.tar.gz
tar -xvzf v0.5.6.tar.gz
cd nequip
pip install . 
```

## Installation and usage
- Download pretrained models.
```bash
git clone git@github.com:hnlab/wB97X-ML.git
cd models
# download all models from https://zenodo.org/records/15514804
pip install zenodo_get
zenodo_get 10.5281/zenodo.15514804 -g "[A-Z]*.tar.gz"
tar -xzvf ./*.tar.gz
```
- Run Energy Prediction Example:  
**Note: Please refer to the corresponding model's README.md for applicable dimer.**

  - Basic (No Multiprocessing)  
 _in Windows/Jupyter environments where `multiprocessing.Pool` is unstable._
```bash
cd scripts
python predict_energy.py -xyz dataset/ACET_ETOH.xyz -md ./models -m ACET -od ./examples
```
  - Parallel Accelerated  
_Leverages `multiprocessing.Pool` for speedup on multicore systems._
```bash
cd scripts
# 2 cores
python predict_energy.py -xyz dataset/ACET_ETOH.xyz -md ./models -m ACET -od ./examples --mlp -w 2
```

Training set: [PDB-FRAGID](https://github.com/hnlab/PDB-FRAGID)  

## Citation
