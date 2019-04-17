"""
Usage:

`python dna.py --output-root datasets/yeast`

This will download a dataset from [Deep Learning Of The Regulatory Grammar
Of Yeast 5′ Untranslated Regions From 500,000 Random Sequences]
(https://genome.cshlp.org/content/27/12/2015) and create the files
`yeast_seq.npy` and `yeast_labels.npy` from it.
These can be uploaded as a dataset in the platform.
"""


from pathlib import Path
import os
import sys

from fire import Fire
import numpy as np
import pandas as pd
from tqdm import tqdm


def one_hot_encoding(df, seq_column, target):
    """
    This function returns a one-hot-encoded representation of DNA sequences
    and a vector of target values from a data frame.

    Args:
        df (DataFrame): Data frame where rows correspond to yeast
            DNA sequences.
        seq_column (str): Name of the column containing the DNA sequence.
        target (str): Name of the column containing the target
            variable (fluorescence).

    Returns:
        X (numpy array): One-hot encoded version of DNA sequence
            with shape (N, 4, 70, 1).
        Y (numpy array: Target value (fluorescence).
        total_width (int): Length of the sequences including padding.
    """

    bases = ["A", "C", "G", "T"]
    base_dict = dict(zip(bases, range(4)))
    n = len(df)

    pad = 10
    total_width = df[seq_column].str.len().max() + 2 * pad

    # initialize an empty numpy ndarray of the appropriate size
    X = np.zeros((n, 4, total_width, 1))

    # an array with the sequences that we will one-hot encode
    seqs = df[seq_column].values

    for i in tqdm(range(n)):
        seq = seqs[i]
        for b in range(len(seq)):
            X[
                i, base_dict[seq[b]], int(b + round((total_width - len(seq)) / 2.0)), 0
            ] = 1.0

    X = X.astype("float32")
    Y = np.asarray(df[target].values, dtype="float32")[:, np.newaxis]

    return X, Y, total_width


def convert(output_root):
    """
    This function downloads a CSV file associated with the paper
    "Deep Learning Of The Regulatory Grammar Of Yeast 5′ Untranslated
    Regions From 500,000 Random Sequences"
    (https://genome.cshlp.org/content/27/12/2015)
    and converts the sequence information and target values to numpy arrays.

    Args:
        output_root (str): Name of directory to which output
            (two numpy array files) will be written.

    Returns:
        -
    """

    output_root = Path(output_root)
    os.makedirs(output_root, exist_ok=True)

    try:
        df = pd.read_csv(
            "https://github.com/animesh/2017---"
            "Deep-learning-yeast-UTRs/blob/master"
            "/Data/Random_UTRs.csv.gz?raw=true",
            compression="gzip",
        )
    except Exception as e:
        sys.exit("Unable to download yeast file ... please check URL")

    df = df.sort_values("t0")
    X, Y, total_width = one_hot_encoding(df, seq_column="UTR", target="growth_rate")

    n = len(df)
    ntrain, nval, ntest = [int(n * x) for x in (0.9, 0.05, 0.05)]
    X_train, Y_train = X[:ntrain], Y[:ntrain]
    X_val, Y_val = X[ntrain : (ntrain + nval)], Y[ntrain : (ntrain + nval)]
    X_test, Y_test = X[(ntrain + nval) :], Y[(ntrain + nval) :]
    X_trainval = np.concatenate([X_train, X_val])
    Y_trainval = np.concatenate([Y_train, Y_val])
    subset_trainval = pd.DataFrame({"subset": ["T"] * ntrain + ["V"] * nval})

    np.save(str(output_root / "yeast_seq_trainval.npy"), X_trainval)
    np.save(str(output_root / "yeast_labels_trainval.npy"), Y_trainval)
    subset_trainval.to_csv(str(output_root / "yeast_subset_trainval.csv"))
    np.save(str(output_root / "yeast_seq_test.npy"), X_test)
    np.save(str(output_root / "yeast_labels_test.npy"), Y_test)


if __name__ == "__main__":
    Fire(convert)
