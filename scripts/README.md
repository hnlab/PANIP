# Active Learning Workflow
Using `PANIP/examples/data_pool` as an example:  
## Step 1: Build Pair Index
For easier lookup, construct an index for each structure file:
```bash
python build_pair_index.py PANIP/examples/data_pool PANIP/examples/pair_file_index.json
```
## Step 2: Initialize Training Set 
Initialize with 2% of the data and use the remaining data for testing:  
```bash
python random_select_frames.py -id PANIP/examples/ -pc 2 -xd PANIP/examples/data_pool/ --init
```
Note: Training data files are overwritten each iteration, and only the most recent iteration is backed up. If you need to preserve training data from every iteration, use `--output` to specify custom filenames.  

## Step 3: Train Model
Train the initial model(s) using the selected dataset.  

## Step 4: Predict and Identify Bad Data  
Run predictions and identify poorly predicted samples.   
Typically, three models are trained in parallel.  

```bash
# You can also use -xyz to batch submit prediction jobs on a cluster
python evaluate_bad_data.py -id PANIP/examples/ -m /path/of/model/ -xd PANIP/examples/data_pool/ -o PANIP/examples/pred/
```
## Step 5: Select Additional Data  
Based on prediction results, select 5% of poorly predicted samples and add them to the training set:
```bash
python random_select_frames.py -id PANIP/examples/ -pc 5 -pred PANIP/examples/pred -xd PANIP/examples/data_pool
```
If you want to judge prediction quality based on the average error across multiple parallel models, you can provide multiple prediction directories after `-pred`.
## Step 6: Iterate Until Convergence  
Retrain the model with the updated training set and repeat the process until the percentage of bad data falls below 5%.  

