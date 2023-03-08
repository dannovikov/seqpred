# Sequences given as an N x L matrix in a csv file
#    Each row repreesnts a sequence as a binary vector
#    1s are mutations from the reference, else 0

# This file creates a pytorch tensor from the csv file

import pandas as pd
import torch

def csv_to_tensor(csv_file):
    df = pd.read_csv(csv_file)
    return torch.tensor(df.values)

