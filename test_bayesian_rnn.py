'''
    Alex: for some unknown reason, the Bayesian RNN doesn't learn
'''

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from scipy.stats import norm
import matplotlib.pyplot as plt
device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
# device = 'cpu'
print(device)
from sklearn.preprocessing import StandardScaler

# from dropout_rnn import DRNN, Optimization, generate_sequence, to_dataframe, inverse_transform, transform_data # Arka, feel free to change these names
from ts_rnn import Model, Optimization, generate_sequence, to_dataframe, inverse_transform, transform_data
from bayesian_rnn import Optimization_BBBRNN
from BBBLayers import BBBRNN


# import datasets
air = pd.read_csv('data/AirQualityUCI.csv', sep=';', skip_blank_lines=True, keep_default_na=True)
# finance = pd.read_csv('data_akbilgic.csv')
flu = pd.read_csv('data/NationalILINet.csv')
# flu = pd.read_csv('data/ILINet.csv')
# rnn_data, rnn_label_wILI = load_flu_data(length, first_year,region)

# for air, index 3 is PT08.S1(CO)
# for flu, index 4 is weighted ILI

dropout_frac = 0.5

# for name, df, col_idx, seq_len in zip(['air','flu'],[air, flu], [3, 4], [100, 52]):
for name, df, col_idx, seq_len in zip(['air'],[air], [3], [100]):
    
    print('=================================================')
    print('===================',name, '===================')
    df.dropna(how='all', inplace=True)
    train_end_idx, val_end_idx = int(df.shape[0]*0.6), int(df.shape[0]*0.8)
    df_train = df.iloc[:train_end_idx, col_idx].to_frame(name='train')  
    df_val = df.iloc[train_end_idx:val_end_idx, col_idx].to_frame(name='val')  
    df_test = df.iloc[val_end_idx:, col_idx].to_frame(name='test')  

    # data scaling
    scaler = StandardScaler()
    train_arr = scaler.fit_transform(df_train)
    val_arr = scaler.transform(df_val)
    test_arr = scaler.transform(df_test)

    x_train, y_train = transform_data(train_arr, seq_len)
    x_val, y_val = transform_data(val_arr, seq_len)
    x_test, y_test = transform_data(test_arr, seq_len)

    # take a look, it's the same sequence but shifted 1
#     print(x_train[0,:10])
#     print(y_train[0,:10])

    # NOTE: change rnn with ours, e.g. rnn=BRNN
    # rnn = nn.LSTMCell.to(device)
    MODE = 'Bayesian'
    # MODE = 'Other'
    if MODE=='Bayesian':
        rnn = BBBRNN # .to(device)
    else:
        rnn = nn.LSTMCell # .to(device)
    # model_1 = DRNN(input_size=1, hidden_size=21, output_size=1, dropout_frac=dropout_frac, rnn=rnn).to(device)
    model_1 = Model(input_size=1, hidden_size=21, output_size=1, dropout_frac=0, rnn=rnn).to(device)
    loss_fn_1 = nn.MSELoss()
    optimizer_1 = optim.Adam(model_1.parameters(), lr=1e-4)
    scheduler_1 = optim.lr_scheduler.StepLR(optimizer_1, step_size=5, gamma=0.1)
    if MODE=='Bayesian':
        optimization_1 = Optimization_BBBRNN(model_1, loss_fn_1, optimizer_1, scheduler_1)
    else:
        optimization_1 = Optimization(model_1, loss_fn_1, optimizer_1, scheduler_1)
    # train with teacher forcing
    optimization_1.train(x_train.to(device), y_train.to(device), x_val.to(device), y_val.to(device), do_teacher_forcing=True, batch_size=32, n_epochs=5)
    # plot loss
    plt.figure()
    # optimization_1.plot_losses()
    
    # evaluation
    actual_1, predicted_1, test_loss_1 = optimization_1.evaluate(x_test, y_test, future=1, batch_size=seq_len)
    df_result_1 = to_dataframe(actual_1, predicted_1) 
    df_result_1 = inverse_transform(scaler, df_result_1, ['actual', 'predicted'])
    plt.figure()
    df_result_1.plot(figsize=(14, 7))
    print("Test loss %.4f" % test_loss_1)



