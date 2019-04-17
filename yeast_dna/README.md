# Yeast DNA regression

This folder contains a script for preprocessing data and evaluating a model on an independent test set for the [Gene expression prediction](https://peltarion.com/knowledge-center/tutorials/gene-expression-prediction) tutorial in the Peltarion Knowledge Center.

The steps to reproduce the results are as follows:

1. Clone this repo and move into the directory in which this README file resides.

2. Execute the `yeast_dna_preprocessing.py` script, indicating where you want the resulting numpy files to be. This script will download the raw sequence data and process them into training, validation and test features and labels in the form of numpy files. For example, if you are happy with the numpy files to be created in the current working directory, you can execute the script like this:

`python yeast_dna_preprocessing.py .`

After a couple of minutes, you should then have the new files `yeast_seq_trainval.npy`, `yeast_labels_trainval.npy`, `yeast_subset_trainval.csv`, `yeast_seq_test.npy` and `yeast_labels_test.npy` in your working directory. The first three files listed above are for training a model in the Peltarion platform, and the last two are an independent test set for evaluating the model outside of the platform.

3. Create and train a model in the Peltarion platform according to the [tutorial](https://peltarion.com/knowledge-center/tutorials/gene-expression-prediction) or really, any way you want. When you create the dataset, you should upload the `yeast_seq_trainval.npy`, which contains the sequence features that will be inputs to the model, the `yeast_labels_trainval.npy`, which will be the labels (targets) for the model, and the `yeast_subset_trainval.csv` CSV file, which indicates which examples should be used for training and validation subsets, respectively. (This is to make the training procedure consistent with the paper we based the tutorial on, rather than doing a random split.)

4. After you have trained the model, create a deployment in the Peltarion platform and enable it.

5. Install the dependencies needed for running the `Yeast DNA Model Evaluation.ipynb` Jupyter notebook (Peltarion's sidekick library, Jupyter, Numpy, Pandas, Seaborn, Scikit-learn) if you don't already have them, perhaps inside a Python virtual environment. 

6. Open the `Yeast DNA Model Evaluation.ipynb` and insert your API credentials (URL and token) in the appropriate place where it says:

```
client = sidekick.Deployment(
    url='<insert deployment URL>',
    token='<insert deployment token>',
    dtypes_in={'seq': 'Numpy (4x70x1)'},
    dtypes_out={'growth_rate': 'Numpy (1)'}
)
```

7. Execute the cells in the notebook.

