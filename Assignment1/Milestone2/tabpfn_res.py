# %%
# !gdown https://drive.google.com/file/d/1Hv4RAltBumSfOkRacoX8qrfDYfd_NDss/view?usp=drive_link --fuzzy

# %%
# !unzip Dataset_AML_Assignment1_Part1.zip

# %%
# from google.colab import drive
# drive.mount('/content/drive')

# %%
import pandas as pd
import numpy as np
import torch

from sklearn.model_selection import train_test_split
from tabpfn import TabPFNClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold

from copy import deepcopy
import gc

import time

import keras.backend as K

import pickle as pkl

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
prefix = './data_dump_tabpfn'

# %%
def accuracy(y_pred, y_test, verbose=False):
    m = y_test.shape[0]
    correct = (y_pred == y_test).sum()
    if verbose:
        print(correct,m)
    accuracy = correct/m
    return accuracy, correct

# %%
def Train_Test(X_train, y_train, all_X_test, class_label, mode, noise_level):
    binary_labels = (y_train == class_label).astype(int)    
    net = TabPFNClassifier(device=device, N_ensemble_configurations=1)
    net.fit(X_train, binary_labels)

    p_test_all = np.array([]) # 32 x b'
    for x_test in all_X_test: #32
        p_test = net.predict_proba(x_test)[:, 1]
        p_test_all = np.concatenate((p_test_all, p_test), axis=None)
    
    # with open(f'{prefix}/net_{mode}_{noise_level}_{str(time.time())}.pkl', 'wb') as f:
    #     pkl.dump(net, f)

    del net
    K.clear_session()
    torch.cuda.empty_cache()
    gc.collect()

    return p_test_all


# %%
def predict(classifiers, X_test):
    predictions = np.zeros((len(classifiers.keys()), X_test.shape[0]))
    for class_label, classifier in classifiers.items():
        p_test = classifier.predict_proba(X_test)[:, 1]
        predictions[class_label, :] = np.array(p_test)
    y_preds = np.argmax(predictions, axis=0)
    return y_preds


# %%
def make_data_splits(df, mode):

    def encode(v, class_values):
        return class_values.index(v)
    
    df = deepcopy(df)

    class_values = df[mode].unique().tolist()
    df[mode] = df[mode].apply(lambda x: encode(x, class_values))
    df.reset_index(drop=True, inplace=True)

    X = df.iloc[:, :26]
    y = df[mode]

    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    return X, y

# %%


# %%
# device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
df_paths = ['../Datasets/df_syn_train_0_0_.csv',
            '../Datasets/df_synA_train_shuffled.csv',
            '../Datasets/df_synA_test_hard_shuffled_sample.csv']

noise_levels = ['none', 'low', 'high']


for i in range(1, 3):
    df = pd.read_csv(df_paths[i])
    print(df.shape)

    
    size_limit = 1_250
    n_splits = (df.shape[0] // size_limit) + 1

    modes = ['era', 'target_5_val', 'target_10_val']

    for mode in modes:
        print("Noise Level:", noise_levels[i], "Mode:", mode)

        all_classes = len(df[mode].unique())

        net_classifiers = []

        all_X_train = []
        all_y_train = []
        all_X_test = []
        all_y_test = []

        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

        
        # jump = 0
        # for batch in range(0, df.shape[0], size_limit):
        #     jump += 1
        #     data = df[batch: batch + size_limit]
        #     X_train, X_test, y_train, y_test = make_data_splits(data, mode)
        #     all_classes = set.union(all_classes, set(y_train.unique()))

        X, y = make_data_splits(df, mode)
        for _, cur_samples_index in skf.split(X, y):
            
            X_cur = np.array(X.iloc[cur_samples_index])
            y_cur = np.array(y.iloc[cur_samples_index])

            X_train, X_test, y_train, y_test = train_test_split(X_cur, y_cur, test_size=0.2, random_state=42, stratify=y_cur)

            all_X_test.append(X_test)
            all_y_test.append(y_test)
            all_X_train.append(X_train)
            all_y_train.append(y_train)
        
        test_correct = 0
        test_samples = 0
        
        all_y_preds = [] #32 x 80,000

        count = 1
        for train_x, train_y in zip(all_X_train, all_y_train): #32
            print(f'{count}/{n_splits}', end = " ")
            count += 1
            y_bin_preds = [] #12 x 80,000
            for class_label in range(all_classes): #12
                y_bin_pred = Train_Test(train_x, train_y, all_X_test, class_label, mode, noise_levels[i]) #32 x b'
                y_bin_preds.append(y_bin_pred) #train on one class, train on some data, test on all samples
            y_bin_preds = np.array(y_bin_preds)
            y_preds = np.argmax(y_bin_preds, axis=0) # 80,000
            all_y_preds.append(y_preds)
        print()
        
        majority_y_preds = []
    
        for sample in range(all_y_preds[0].shape[0]):
            voting_array = [0] * all_classes
            for batch in range(len(all_y_preds)):
                voting_array[all_y_preds[batch][sample]] += 1
            max_val = -1
            max_ind = -1
            for ind in range(len(voting_array)):
                if voting_array[ind] > max_val:
                    max_val = voting_array[ind]
                    max_ind = ind
            majority_y_preds.append(max_ind)
        
        majority_y_preds = np.array(majority_y_preds)
        print("Accuracy:", accuracy(majority_y_preds, np.array(all_y_test).reshape((-1,)))[0])
    

