import pandas as pd
import numpy as np
import torch
from torch import nn
from torch import optim
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn.functional as F
from IPython import display
from tqdm.notebook import tqdm
import random
import math, time, os
from matplotlib import pyplot as plt
import pickle as pkl

from AutoEncoder import Autoencoder as AE
from copy import deepcopy
from sklearn.model_selection import train_test_split

device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
prefix = './data_dump'


class SinusodialDataset(Dataset):
    def __init__(self, df, test=False):
        """ creating label columns of eras and targets """
        self.NUM_FEATURES = 24
        self.X = df.iloc[:, :self.NUM_FEATURES]
        self.testMode = test
        if self.testMode == False:
            self.y = df['target_10_val']
        self.X = self.create_categorical_one_hot(self.X)
    
    def create_categorical_one_hot(self, df):
        categories = [0, 0.25, 0.5, 0.75, 1]
        one_hot_encoded_columns = []
        for col in df.columns:
            for cat in categories:
                new_col_name = f"{col}_{cat}"
                one_hot_encoded_col = (df[col] == cat).astype(int)
                one_hot_encoded_col.name = new_col_name
                one_hot_encoded_columns.append(one_hot_encoded_col)

        # Concatenate the one-hot encoded columns along axis 1
        return pd.concat(one_hot_encoded_columns, axis=1)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        X = torch.tensor(self.X.iloc[idx].values, dtype=torch.float32)
        if self.testMode == False:
            y = torch.tensor(self.y.iloc[idx], dtype=torch.long)
            return X, y
        return X
    

def make_data_splits(train_df, val_df, test_df, batch_size=32):

    def encode(v, class_values):
        return class_values.index(v)

    class_values = train_df['target_10_val'].unique().tolist()
    train_df['target_10_val'] = train_df['target_10_val'].apply(lambda x: encode(x, class_values))
    train_df.reset_index(drop=True, inplace=True)

    data_train = SinusodialDataset(train_df)
    loader_train = DataLoader(data_train, batch_size=batch_size, shuffle=False)
    loader_val = None
    loader_test = None

    if val_df is not None:
        class_values = val_df['target_10_val'].unique().tolist()
        val_df['target_10_val'] = val_df['target_10_val'].apply(lambda x: encode(x, class_values))
        val_df.reset_index(drop=True, inplace=True)

        data_val = SinusodialDataset(val_df)
        loader_val = DataLoader(data_val, batch_size=batch_size, shuffle=False)

    if test_df is not None:
        class_values = test_df['target_10_val'].unique().tolist()
        test_df['target_10_val'] = test_df['target_10_val'].apply(lambda x: encode(x, class_values))
        test_df.reset_index(drop=True, inplace=True)
            
        data_test = SinusodialDataset(test_df)
        loader_test = DataLoader(data_test, batch_size=batch_size, shuffle=False)

    print("Train-val-test lengths: ", len(data_train), len(data_val), len(data_test))

    return loader_train, loader_val, loader_test

class TestTimeAdapter(nn.Module):
    
    def __init__(self, ae_dims, cl_dims, lr=1e-3, weight_decay=1e-3):
        super(TestTimeAdapter,self).__init__()
        self.ae_dims=ae_dims
        self.cl_dims = cl_dims

        self.ae = AE(ae_dims)
        self.classifier=nn.ModuleList()
        
        for i in range(len(cl_dims)-2):
            self.classifier.append(nn.Linear(cl_dims[i],cl_dims[i+1]))
            self.classifier.append(nn.ReLU())
        self.classifier.append(nn.Linear(cl_dims[i+1],cl_dims[i+2]))
        self.classifier.append(nn.LogSoftmax(dim=1))
        self.optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)

    def forward(self, x, classifier=False):
        if classifier:
            x = x.float()
            x = self.ae.encode(x)
            for l in self.classifier:
                x = l(x)
            return x
        
        x = x.float()
        x = self.ae(x)
        return x
    
def accuracy(y_pred, y_test, verbose=True):
    m = y_test.shape[0]
    predicted = torch.max(y_pred, 1)[1]
    correct = (predicted == y_test).float().sum().item()
    if verbose:
        print(correct,m)
    accuracy = correct/m
    return accuracy, correct

def Test(net, loader_test, \
         device='cpu', Loss=nn.NLLLoss(reduction='sum'), \
         test_time_epochs = 3):
    net.train()
    total_samples = 0
    correct_samples = 0
    loss = 0.0
    step=0
    (X_prev, y_prev) = (None, None)
    (X_prev_prev, y_prev_prev) = (None, None)
    # test_mode is False
    for (X, y) in loader_test:
        # print(step, end=" ")
        X=X.to(device)
        y=y.to(device)
        total_samples += y.shape[0]
   
        for e in range(test_time_epochs): 
            x_reconst = net(X, classifier=False)
            ae_loss = 0.0
            for feat in range(0, X.shape[-1], 5):
                ae_loss += Loss(nn.LogSoftmax(dim=1)(x_reconst[:, feat:feat+5]),X[:, feat:feat+5].argmax(dim=1))
            net.optimizer.zero_grad()
            ae_loss.backward()
            net.optimizer.step()
        
            if X_prev_prev is not None:
                y_pred = net(X_prev_prev, classifier=True)
                cl_loss = Loss(y_pred, y_prev_prev)
                net.optimizer.zero_grad()
                cl_loss.backward()
                net.optimizer.step()

        y_pred = net(X, classifier=True)
        cl_loss = Loss(y_pred, y)

        loss += (ae_loss+cl_loss).item()
        _, i_cor_sam = accuracy(y_pred, y,verbose=False)
        correct_samples += i_cor_sam
        step+=1

        (X_prev_prev, y_prev_prev) = (X_prev, y_prev)
        (X_prev, y_prev) = (X, y)

    # print()
    
    acc = correct_samples / total_samples
    loss /= total_samples
    print('Test/Val loss:', loss, 'Test/Val acc:', acc)
    return loss, acc

def Train(Net, train_loader, epochs=20, Loss=nn.NLLLoss(reduction='sum'), 
          verbose=False, device='cpu',
          val_ds=None, loader_test=None):
    model_save_time = time.time()
    losses = []
    accs = []
    val_losses=[]
    val_accL=[]
    Net.to(device)
    for e in range(epochs):
        Net.train()
        step=0
        tot_loss=0.0
        start_time = time.time()
        correct_samples = 0
        total_samples = 0
        # test_mode = False
        for (X,y) in train_loader:
            X=X.to(device)
            y=y.to(device)
            total_samples += y.shape[0]
            x_reconst = Net(X, classifier=False) # B x nd x nc = B x 24 x 5
            ae_loss = 0.0
            for feat in range(0, X.shape[-1], 5):
                # print((x_reconst[:, feat:feat+5]),X[:, feat:feat+5].argmax(dim=1))
                # print((x_reconst[:, feat:feat+5]).shape,X[:, feat:feat+5].argmax(dim=1).shape)
                ae_loss += Loss(nn.LogSoftmax(dim=1)(x_reconst[:, feat:feat+5]),X[:, feat:feat+5].argmax(dim=1))
            Net.optimizer.zero_grad()
            ae_loss.backward()
            Net.optimizer.step()

            y_pred = Net(X, classifier=True)
            cl_loss = Loss(y_pred, y)
            Net.optimizer.zero_grad()
            cl_loss.backward()
            Net.optimizer.step()

            step+=1
            tot_loss+=(ae_loss+cl_loss)
            if verbose:
                _, i_cor_sam = accuracy(y_pred, y,verbose=False)
                correct_samples += i_cor_sam
            
        end_time = time.time()
        t = end_time-start_time
        l = tot_loss.item()/total_samples
        losses += [l]
        a = correct_samples/total_samples
        accs += [a]

        if verbose:
            print('Epoch %2d Loss: %2.5e Accuracy: %2.5f Epoch Time: %2.5f' %(e,l,a,t))

        val_loss, val_acc = Test(deepcopy(Net), val_ds, device = device)
        val_losses.append(val_loss)
        val_accL.append(val_acc)

        # print("TESTING BUDDY:")
        # Test(deepcopy(Net), loader_test, device=device)

        torch.save(Net.state_dict(), f'{prefix}/net_{str(model_save_time)}.pth')

    return Net, losses, accs, val_losses, val_accL


def plot_loss_acc(losses, accs, val_losses, val_accs):

    plt.plot(np.array(accs),color='red', label='Train accuracy')
    plt.plot(np.array(val_accs),color='blue', label='Val accuracy')
    plt.legend()
    plt.savefig(f'{prefix}/acc_tta.png')
    plt.clf()

    plt.plot(np.array(losses),color='red', label='Train loss')
    plt.plot(np.array(val_losses),color='blue', label='Val loss')
    plt.legend()
    plt.savefig(f'{prefix}/loss_TTA.png')
    plt.clf()
    return

df_paths = ['../Datasets/df_train_shuffled.csv', 
              '../Datasets/df_val.csv',
              '../Datasets/df_val_test.csv']

batch_size = 140

train_df = pd.read_csv(df_paths[0])
val_df = pd.read_csv(df_paths[1])
test_df = pd.read_csv(df_paths[2])

loader_train, loader_val, loader_test = make_data_splits(train_df, val_df, test_df, batch_size=batch_size)
net = TestTimeAdapter(ae_dims=[120, 64, 32, 16, 32, 64, 120], 
                        cl_dims=[16, 16, 5], ).to(device)
net, losses, accs, val_losses, val_accL = Train(net, loader_train, \
                                epochs=5, verbose=True, device=device, \
                                    val_ds=loader_val,loader_test=loader_test)
plot_loss_acc(losses, accs, val_losses, val_accL)
#Testing code
test_loss, test_acc = Test(deepcopy(net), loader_test, device=device)
print(test_loss, test_acc)


def TestTime(net, loader_test, \
         device='cpu', Loss=nn.NLLLoss(reduction='sum'), \
         test_time_epochs = 3):
    net.train()
    total_samples = 0
    loss = 0.0
    step=0
    
    # test_mode is True
    for X in loader_test:
        X=X.to(device)
        
        total_samples += X.shape[0]
   
        for e in range(test_time_epochs): 
            x_reconst = net(X, classifier=False)
            ae_loss = 0.0
            for feat in range(0, X.shape[-1], 5):
                ae_loss += Loss(nn.LogSoftmax(dim=1)(x_reconst[:, feat:feat+5]),X[:, feat:feat+5].argmax(dim=1))
            net.optimizer.zero_grad()
            ae_loss.backward()
            net.optimizer.step()

        loss += (ae_loss).item()
        step+=1
    
    loss /= total_samples
    print('Test/Val loss:', loss)
    return loss


import os

def encode(v, class_values):
        return class_values.index(v)


def list_csv_files(directory):
    csv_files = [file for file in os.listdir(directory) if file.endswith('.csv')]
    return csv_files

for date in [2,3,4,5,8]:
    file_path = f'../Datasets/live_data_0{date}-Apr-2024'
    csv_files = list_csv_files(file_path)

    round_dict = {}

    for i in range(len(csv_files)):
        round = int(csv_files[i].split(".")[0].split("_")[-1])
        if round not in round_dict.keys():
            round_dict[round] = {'train': '', 'test': ''}
        if 'train' in csv_files[i]:
            round_dict[round]['train'] = file_path + '/' + csv_files[i]
        else:
            round_dict[round]['test'] = file_path + '/' + csv_files[i]
            

    results = pd.DataFrame()

    for round in round_dict.keys():
        print('Working on - ', round_dict[round]['train'])
        df = pd.read_csv(round_dict[round]['train'])
        class_values = df['target_10_val'].unique().tolist()
        df['target_10_val'] = df['target_10_val'].apply(lambda x: encode(x, class_values))
        df.reset_index(drop=True, inplace=True)
        data = SinusodialDataset(df, test=False)
        loader_adapt = DataLoader(data, batch_size=1, shuffle=False)
        adapt_loss, adapt_acc = Test(net, loader_adapt, device=device, test_time_epochs=2)
        
        print('Working on - ', round_dict[round]['test'])
        df = pd.read_csv(round_dict[round]['test'])

        ids = df['id']
        df = df.drop(columns=['id'])
        row_num = df['row_num']
        
        data = SinusodialDataset(df, test=True)
        loader_adapt = DataLoader(data, batch_size=1, shuffle=False)
        adapt_loss = TestTime(net, loader_adapt, device=device)

        predictions = []

        total_samples = 0
        for X in loader_adapt:
            X=X.to(device)
            total_samples += X.shape[0]
            y_pred = net(X, classifier=True)
            predictions.append(y_pred.argmax(dim=1).detach().cpu().numpy())
            print(predictions)

        temp_df = pd.DataFrame()
        temp_df['id'] = ids
        temp_df['predictions'] = predictions
        temp_df['row_num'] = row_num
        temp_df['round_no'] = [f'{round}']*ids.shape[0]
        results = pd.concat([results, temp_df])
    
    
    results.to_csv(f'predictions_0{date}-04-2024.csv')
