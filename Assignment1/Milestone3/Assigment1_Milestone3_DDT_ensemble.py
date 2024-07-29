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


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
prefix = './data_dump_selective_classification/'


# %%
# NAL - ninju
# Ensemble - mudit
# Autoencoder - ninju
# SubTab
# Meta Learning (clean+dirty)

# %%
class SinusodialDataset(Dataset):
    def __init__(self, df, mode='era'):
        """ creating label columns of eras and targets """
        self.X = df.iloc[:, :24]
        if mode == 'era':
          self.y = df['era_label']
        else:
          self.y = df[f'{mode}']

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        X = torch.tensor(self.X.iloc[idx].values, dtype=torch.float32)
        y = torch.tensor(int(self.y.iloc[idx]), dtype=torch.long)
        return X, y

# %%
def make_data_splits(df, mode, batch_size=32, train_perc=0.7, val_test_perc=0.5):

    def encode(v, class_values):
        return class_values.index(v)

    #adding new era_label column indexed 0, 1,...
    class_values = df['era'].unique().tolist()
    df['era_label'] = df['era'].apply(lambda x: encode(x, class_values))
    df.reset_index(drop=True, inplace=True)

    train_samples = int(len(df)*train_perc)
    val_test_samples = len(df)-train_samples

    data = SinusodialDataset(df, mode=mode)
    data_train, data_test = random_split(data, [train_samples, val_test_samples])

    val_samples = int(len(data_test)*0.5)
    test_samples = len(data_test)-val_samples
    data_val, data_test = random_split(data_test, [val_samples, test_samples])

    print("Train-val-test lengths: ", len(data_train), len(data_val), len(data_test))

    loader_train = DataLoader(data_train, batch_size=batch_size, shuffle=True)
    loader_val = DataLoader(data_val, batch_size=batch_size, shuffle=False)
    loader_test = DataLoader(data_test, batch_size=batch_size, shuffle=False)

    return loader_train, loader_val, loader_test, data

# %% [markdown]
# #DDT

# %%
class DifferentiableDecisionTree(nn.Module):
    def __init__(self, input_dim, output_dim, max_depth=3):
        super(DifferentiableDecisionTree, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.max_depth = max_depth

        # Parameters for decision rules
        self.feature_threshold = nn.Parameter(torch.randn(max_depth, input_dim))
        self.left_weights = nn.Parameter(torch.randn(max_depth, output_dim, input_dim))
        self.right_weights = nn.Parameter(torch.randn(max_depth, output_dim, input_dim))

        self.optimizer = optim.SGD(self.parameters(), lr=0.01)


    def forward(self, x):
        # Initialize leaf node values to zero
        leaf_values = torch.zeros(x.shape[0], self.output_dim).to(device)

        for d in range(self.max_depth):
            # Compute decision rule
            decision = x[:, None, :] < self.feature_threshold[d]
            # Apply decision rule to update leaf node values
            left_values = torch.mul(decision.float(), self.left_weights[d]).sum(dim=-1)
            right_values = torch.mul((1 - decision.float()), self.right_weights[d]).sum(dim=-1)
            leaf_values = leaf_values + left_values + right_values

        return leaf_values



# %%
def accuracy(Net, X_test, y_test, verbose=True):
    Net.eval()
    m = X_test.shape[0]
    y_pred = Net(X_test)
    predicted = torch.max(y_pred, 1)[1]
    correct = (predicted == y_test).float().sum().item()
    if verbose:
        print(correct,m)
    accuracy = correct/m
    Net.train()
    return accuracy, correct

# %%
def Test(net, loader_test, mode, noise_level, \
         device='cpu', Loss=nn.CrossEntropyLoss(reduction='sum')):
    net.eval()
    total_samples = 0
    correct_samples = 0
    loss = 0.0
    for (X, y) in loader_test:
        X=X.to(device)
        y=y.to(device)
        total_samples += y.shape[0]
        _, i_cor_sam = accuracy(net,X,y,verbose=False)
        correct_samples += i_cor_sam
        loss += Loss(net(X), y).cpu().detach().item()
    acc = correct_samples / total_samples
    loss /= total_samples
    return loss, acc

# %%
def Train(Net, data, mode, noise_level, epochs=20, lr=5e-2, Loss=nn.CrossEntropyLoss(reduction='sum'), verbose=False, device='cpu',
          val_ds=None, plot_accs=False, plot_losses=False):
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
        for (X,y) in data:
            X=X.to(device)
            y=y.to(device)
            total_samples += y.shape[0]
            y_pred = Net(X)
            loss = Loss(y_pred,y)
            Net.optimizer.zero_grad()
            loss.backward()
            Net.optimizer.step()
            step+=1
            tot_loss+=loss
            if verbose:
                _, i_cor_sam = accuracy(Net,X,y,verbose=False)
                correct_samples += i_cor_sam
        end_time = time.time()
        t = end_time-start_time
        l = tot_loss.item()/total_samples
        losses += [l]
        if verbose:
            a = correct_samples/total_samples
            accs += [a]
            print('Epoch %2d Loss: %2.5e Accuracy: %2.5f Epoch Time: %2.5f' %(e,l,a,t))

        val_loss, val_acc = Test(Net, val_ds, mode, noise_level, device)
        val_losses.append(val_loss)
        val_accL.append(val_acc)

        torch.save(Net.state_dict(), f'{prefix}/net_{noise_level}_{mode}_{str(model_save_time)}.pth')

    return Net, losses, accs, val_losses, val_accL

# %%
def plot_loss_acc(losses, accs, val_losses, val_accs, mode, noise_level):

    plt.plot(np.array(accs),color='red', label='Train accuracy')
    plt.plot(np.array(val_accs),color='blue', label='Val accuracy')
    plt.legend()
    plt.savefig(f'{prefix}/acc_{mode}_{noise_level}.png')
    plt.clf()

    plt.plot(np.array(losses),color='red', label='Train loss')
    plt.plot(np.array(val_losses),color='blue', label='Val loss')
    plt.legend()
    plt.savefig(f'{prefix}/loss_{mode}_{noise_level}.png')
    plt.clf()
    return

def checkModel(net, mode, noise_level, loader_train, loader_val, loader_test):
  net, losses, accs, val_losses, val_accL = Train(net, loader_train, mode, noise_level, \
                              epochs=15, verbose=True, device=device, val_ds=loader_val, \
                              plot_accs=True, plot_losses=True)

  plot_loss_acc(losses, accs, val_losses, val_accL, mode, noise_level)
  #Testing code
  test_loss, test_acc = Test(net, loader_test, mode, noise_level, device=device)
  return net, accs[-1]

def TestEnsemble(nets, loader_test, mode, noise_level, \
         device='cpu', Loss=nn.CrossEntropyLoss(reduction='sum')):
    total_samples = 0
    correct_samples = 0

    for (X, y) in loader_test:
        X=X.to(device)
        y=y.to(device)
        sample_size = y.shape[0]
        total_samples += sample_size

        y_preds = []
        all_classes = 0

        for net in nets:
          net.eval()
          __y_pred = net(X)
          all_classes = __y_pred.shape[-1]
          predicted = torch.max(__y_pred, 1)[1]
          y_preds.append(predicted.cpu().detach().numpy())


        predicted = []

        for i in range(sample_size):
          voting_array = [0] * all_classes
          for model_num in range(len(nets)):
            voting_array[y_preds[model_num][i]] += 1
          max_val = -1
          max_ind = -1
          for ind in range(len(voting_array)):
            if voting_array[ind] > max_val:
              max_val = voting_array[ind]
              max_ind = ind

          predicted.append(max_ind)

        predicted = torch.tensor(predicted).to(device)
        i_cor_sam = (predicted == y).float().sum().item()
        correct_samples += i_cor_sam

    acc = correct_samples / total_samples
    return acc

# %%
df_paths = ['../Datasets/df_syn_train_0_0_.csv',
            '../Datasets/df_synA_train_shuffled.csv',
            '../Datasets/df_synA_test_hard_shuffled_sample.csv']

noise_levels = ['none', 'low', 'high']
batch_sizes = [32, 128, 128]

losses_arr = []
accs_arr = []

for i in range(0, 3):
    df = pd.read_csv(df_paths[i])

    modes = ['era', 'target_5_val', 'target_10_val']

    loss_per_mode = []
    acc_per_mode = []

    for mode in modes:
        noise_level = noise_levels[i]
        print("Noise Level:", noise_levels[i], "Mode:", mode)
        loader_train, loader_val, loader_test, data = make_data_splits(df, mode=mode, \
                                                                       batch_size=batch_sizes[i])
        # net = MLP(dims=[data.X.shape[1], 32, 64, 32, len(data.y.unique())]).to(device)
        nets = [DifferentiableDecisionTree(input_dim=data.X.shape[1], output_dim=len(data.y.unique()), max_depth=3).to(device),
                DifferentiableDecisionTree(input_dim=data.X.shape[1], output_dim=len(data.y.unique()), max_depth=5).to(device),
                DifferentiableDecisionTree(input_dim=data.X.shape[1], output_dim=len(data.y.unique()), max_depth=1).to(device)
                ]

        mode += '_ensemble'
        check_accs = []
        counter = 0
        for net in nets:
            net = net.to(device)
            net, _acc = checkModel(net, mode, noise_level, loader_train, loader_val, loader_test)
            check_accs.append(_acc)
            # print(f'Model {counter} finished - {_acc}%')
            counter += 1

        # for j in range(len(nets)):
        # print(print(f'Model {j} accuracy - {check_accs[j]}%'))

        ensemble_acc = TestEnsemble(nets, loader_test, mode, noise_level, device=device)
        print(f'"Noise Level: {noise_level} | Mode: {mode} | Ensemble model acc: {ensemble_acc}')

        # loss_per_mode.append(test_loss)
        acc_per_mode.append(ensemble_acc)

    # losses_arr.append(loss_per_mode)
    accs_arr.append(acc_per_mode)

# %%
with open(f'{prefix}/losses_dump.pkl', 'wb') as f:
    pkl.dump(losses_arr, f)

with open(f'{prefix}/accs_dump.pkl', 'wb') as f:
    pkl.dump(accs_arr, f)

# %%



