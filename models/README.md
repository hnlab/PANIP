# MODELS

## Download
```bash
# download all models from https://zenodo.org/records/15514804
pip install zenodo_get
zenodo_get 10.5281/zenodo.15514804 -g "[A-Z]*.tar.gz"
tar -xzvf ./*.tar.gz
# download specific model
# for example, ACEM model ensemble
wget https://zenodo.org/record/15514804/files/ACEM.tar.gz
tar -xzvf ACEM.tar.gz
```
## Model Nameing Convention
- `[Fragment]`: Specialized for specific molecular fragments. See `README.md` under each model directory for more details.
- `GLOBAL`: Trained on all fragment types for universal prediction.  

## Fragment Types

![monomers_1line](https://github.com/user-attachments/assets/545660ab-3196-46b7-b167-d92df9422b4c)  
