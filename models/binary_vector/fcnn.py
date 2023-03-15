# This is a baseline model that uses a fully connected network on the task of variant prediction

# Data is N sequences of length L given as an N x L matrix/
# Sequences are sorted by date 
# target is 1 x L vector

# A call to a forward function will be on the set of sequences in the sliding windows
# since we are getting a set of vectors as input, let's flatten them.





import torch
import torch.nn as nn
import torch.nn.functional as F


class FCNN(nn.Module):
    def __init__(self, N, L):
        """
        N: number of sequences in the sliding window
        L: length of each sequence
        """
        super().__init__()
        self.N = N
        self.L = L
        flat_size = N*L
        self.fc1 = nn.Linear(flat_size, flat_size//8)
        self.fc2 = nn.Linear(flat_size//8, flat_size//16)
        self.fc3 = nn.Linear(flat_size//16, L)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x is a N x L Tensor 
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x





