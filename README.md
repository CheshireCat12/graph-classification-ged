# Graph Classification with GED and KNN

This python module classifies graphs using GED and KNN as the classifiers.
It first computes the distances between all the graphs in the given datasets for all the alphas in the list and saves them for later use.
Then, it performs k-fold cross validation n times.
The classification accuracies are saved for each trial.

## Install

First, clone the main repository and 'graph-matching-core' repository.
Then, install numpy and the 'graph-matching-core' package by running the appropriate commands in your terminal or command prompt.
```bash
# Clone the current repo
git clone https://github.com/CheshireCat12/graph-classification-ged.git
cd graph-classification-ged

# Clone the 'graph-matching-core' repo
git clone https://github.com/CheshireCat12/graph-matching-core.git

# Install the required packages
pip install numpy
pip install -e graph-matching-core
pip install -r requirements.txt
```

## How to use

### General Command

```bash
usage: main.py [-h] --root_dataset ROOT_DATASET [--parameters_edit_cost PARAMETERS_EDIT_COST [PARAMETERS_EDIT_COST ...]] [--alphas [ALPHAS ...]] [--ks [KS ...]]
               [--n_trials N_TRIALS] [--n_outer_cv N_OUTER_CV] [--n_inner_cv N_INNER_CV] [--n_cores N_CORES] [--save_gt_labels] [--save_predictions] --folder_results
               FOLDER_RESULTS [-v]
               {} ...

Graph Classification Using KNN with GED

positional arguments:
  {}

optional arguments:
  -h, --help            show this help message and exit
  --root_dataset ROOT_DATASET
                        Root of the dataset
  --parameters_edit_cost PARAMETERS_EDIT_COST [PARAMETERS_EDIT_COST ...]
                        Tuple with the cost for the edit operations
  --alphas [ALPHAS ...]
                        List of alphas to test
  --ks [KS ...]         List of ks to test (k being the number of neighbors for the KNN)
  --n_trials N_TRIALS   Number of cross-validation to perform
  --n_outer_cv N_OUTER_CV
                        Number of outer loops in the cross-validation
  --n_inner_cv N_INNER_CV
                        Number of inner loops in the cross-validation
  --n_cores N_CORES     Set the number of cores to use.If n_cores == 0 then it is run without parallelization.If n_cores > 0 then use this number of cores
  --save_gt_labels      save the ground truth classes if activated
  --save_predictions    save the predicted classes if activated
  --folder_results FOLDER_RESULTS
                        Folder where to save the classification results
  -v, --verbose         Activate verbose print
```

### Example

Run the following line to download and generate the Enzyme dataset.

```bash
python main.py --root_dataset ./data/enzymes --alphas 0.5 0.6 0.7 --ks 7 11 --n_trials 10 --n_outer_cv 10 --n_inner_cv 5 --n_cores 10 --save_predictions --save_gt_labels --folder_results ./results/enzymes -v

```
