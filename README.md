# PAirwise Non-covalent Interaction Potential model (PANIP)
Robust machine learning interatomic potentials (MLIPs) that achieve accuracy comparable to the ωB97X-D3BJ/def2-TZVPP quantum mechanical method on non-covalent interactions.  

---

## Requirements
PANIP is built on NequIP, please install NequIP first.   
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
git clone git@github.com:hnlab/PANIP.git
cd models
# download all models from https://zenodo.org/records/15514804
pip install zenodo_get
zenodo_get 10.5281/zenodo.18213084
tar -xzvf ./*.tar.gz
```
- Run Energy Prediction Example:  
**Note: Please refer to the corresponding model's `README.md` for applicable dimer.**

  - Basic (No Multiprocessing)  
 _in Windows/Jupyter environments where `multiprocessing.Pool` is unstable._
  ```bash
  cd scripts
  # use global model
  python predict_energy.py -xyz examples/ACET_ETOH.xyz -md ./models -m GLOBAL -od ./examples
  # use sepecific model
  python predict_energy.py -xyz examples/ACET_ETOH.xyz -md ./models/split_models -m ACET -od ./examples
  ```

  - Parallel Accelerated  
_Leverages `multiprocessing.Pool` for speedup on multicore systems._
  ```bash
  cd scripts
  # 2 cores
  python predict_energy.py -xyz examples/ACET_ETOH.xyz -md ./models/split_models -m ACET -od ./examples --mlp -w 2
  ```

Training set: [PDB-FRAGID](https://github.com/hnlab/PDB-FRAGID)  

## Citation
[Developing a Machine-Learning Interatomic Potential for Non-Covalent Interactions in Proteins](https://arxiv.org/submit/7154354/view)  
