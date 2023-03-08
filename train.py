from models.binary_vector.fcnn import FCNN
from preproc.csv_to_tensor import csv_to_tensor

import torch
import torch.nn as nn
import torch.nn.functional as F

import pandas as pd
import numpy as np
import os
from matplotlib import pyplot as plt

from tqdm import tqdm

#data paths
# DATA_PATH = "./data/USA/usa_matrix.csv"
DATA_PATH = r"data\USA\usa_50k_matrix.csv"
DATA_FORMAT = "csv"
DATES_PATH = r"data\USA\usa_int_dates_50k.csv"
DATES_FORMAT = "int"

#training hyperparameters
EPOCHS = 10
LEARNING_RATE = 0.001
BATCH_SIZE = 64
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# WINDOW_SIZE_DAYS = 90 #number of days in the sliding window
# TARGET_SIZE_DAYS = 14 #number of days ahead on which to compute consensus target
WINDOW_SIZE= 1000 #number of sequences in the sliding window
TARGET_SIZE = 100 #number of sequences ahead on which to compute consensus target

#Precompouted pickle paths
TRAIN_SLIDES_PATH =  "./pickles/usa50k_train_slides.pkl"
TRAIN_TARGETS_PATH = "./pickles/usa50k_train_targets.pkl"
TEST_SLIDES_PATH =   "./pickles/usa50k_test_slides.pkl"
TEST_TARGETS_PATH =  "./pickles/usa50k_test_targets.pkl"
VAL_SLIDES_PATH =    "./pickles/usa50k_val_slides.pkl"
VAL_TARGETS_PATH =   "./pickles/usa50k_val_targets.pkl"

def main():
    print('Reading data')
    data = readData()
    print('Creating sorted tensors')
    train_data, test_data, val_data, *dates = createSortedTensors(data)
    print('Creating data loaders')
    train_loader, test_loader, val_loader = createDataLoaders(train_data, test_data, val_data)
    print('Training model')
    sequence_length = train_data.shape[1]
    model = FCNN(WINDOW_SIZE, sequence_length)
    training_losses = train(model, train_loader, test_loader)
    eval_loss = evaluate(model, val_loader)

    plt.plot(training_losses)
    plt.show()

    print(f"Eval loss: {eval_loss}")


def readData():
    if DATA_FORMAT == "csv":
        data = csv_to_tensor(DATA_PATH)
        return data
    raise Exception("Data format not supported")

    
def createSortedTensors(data):
    if DATES_FORMAT == "int":
        dates = pd.read_csv(DATES_PATH, header=None)
    elif DATES_FORMAT == "date":
        dates = pd.read_csv(DATES_PATH, parse_dates=True, header=None)


    dates = dates.values
    dates = dates.reshape(-1)
    sorting_indices = dates.argsort()
    print(f"{dates=}, {sorting_indices=}, {data.shape=}, {len(dates)=}")

    data = data[sorting_indices]

    train_data = data[:int(0.6*len(data))]
    test_data = data[int(0.6*len(data)):int(0.9*len(data))]
    val_data = data[int(0.9*len(data)):]

    dates.sort()
    train_dates = dates[:int(0.6*len(dates))]
    test_dates = dates[int(0.6*len(dates)):int(0.9*len(dates))]
    val_dates = dates[int(0.9*len(dates)):]

    return train_data, test_data, val_data, train_dates, test_dates, val_dates


def createDataLoaders(train_data, test_data, val_data):

    if not os.path.exists(TRAIN_SLIDES_PATH):
        train_slides, train_targets = createSlidingWindowSet(train_data)
        pickle(train_slides, TRAIN_SLIDES_PATH)
        pickle(train_targets, TRAIN_TARGETS_PATH)
    else:
        train_slides = unpickle(TRAIN_SLIDES_PATH)
        train_targets = unpickle(TRAIN_TARGETS_PATH)

    if not os.path.exists(TEST_SLIDES_PATH):
        test_slides, test_targets = createSlidingWindowSet(test_data)
        pickle(test_slides, TEST_SLIDES_PATH)
        pickle(test_targets, TEST_TARGETS_PATH)
    else:
        test_slides = unpickle(TEST_SLIDES_PATH)
        test_targets = unpickle(TEST_TARGETS_PATH)

    if not os.path.exists(VAL_SLIDES_PATH):
        val_slides, val_targets = createSlidingWindowSet(val_data)
        pickle(val_slides, VAL_SLIDES_PATH)
        pickle(val_targets, VAL_TARGETS_PATH)
    else:
        val_slides = unpickle(VAL_SLIDES_PATH)
        val_targets = unpickle(VAL_TARGETS_PATH)


    train_ds = torch.utils.data.TensorDataset(train_slides, train_targets)
    test_ds = torch.utils.data.TensorDataset(test_slides, test_targets)
    val_ds = torch.utils.data.TensorDataset(val_slides, val_targets)

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=True)

    return train_loader, test_loader, val_loader


def createSlidingWindowSet(data):
    slides = []
    targets = []
    print(len(data) - WINDOW_SIZE - TARGET_SIZE, "windows to create")
    for i in tqdm(range(len(data) - WINDOW_SIZE - TARGET_SIZE)):
        input_window = data[i:i+WINDOW_SIZE]
        target_window = data[i+WINDOW_SIZE:i+WINDOW_SIZE+TARGET_SIZE]
        target_consensus = get_consensus(target_window)
        slides.append(input_window)
        targets.append(target_consensus)
        
    return torch.stack(slides), torch.stack(targets)


def get_consensus(target_window):
    consensus = torch.zeros(target_window.shape[1])
    for i in range(target_window.shape[1]):
        col = target_window[:, i]
        counts = torch.bincount(col)
        consensus[i] = torch.argmax(counts)

    return consensus


def train(model, train_loader, test_loader):
    model.to(DEVICE)

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    training_losses = []
    for epoch in tqdm(range(EPOCHS)):
        model.train()
        for i, (slides, targets) in tqdm(enumerate(train_loader), leave=False):
            slides = slides.to(DEVICE)
            targets = targets.to(DEVICE)
            optimizer.zero_grad()
            output = model(slides)
            loss = criterion(output, targets)
            loss.backward()
            optimizer.step()

            if i % 10 == 0:
                print(f"Epoch: {epoch}, Batch: {i}, Loss: {loss.item()}")
                training_losses.append(loss.item())

        test_loss = evaluate(model, test_loader)
        print(f"Epoch: {epoch}, Test Loss: {test_loss}")
        training_losses.append(test_loss)

    return training_losses
        

def evaluate(model, val_loader):
    model.to(DEVICE)
    model.eval()
    criterion = nn.BCELoss()
    with torch.no_grad():
        for i, (slides, targets) in enumerate(val_loader):
            slides = slides.to(DEVICE)
            targets = targets.to(DEVICE)
            output = model(slides)
            loss = criterion(output, targets)

    return loss.item()

def pickle(obj, path):
    if os.path.exists(path):
        return
    with open(path, "wb") as f:
        pickle.dump(obj, f)

def unpickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)

if __name__ == "__main__":
    main()