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


def uncertainty_estimate(x, model, num_samples, future, dropout_rate, decay, N, l2=0.01):
    x = x.to(device)
#         print(x.shape)
#         print(self.model(x, future=future)[:,-future:].cpu().detach().numpy())
#     print(model(x, future=future)[:,-future:].cpu().detach().numpy().shape)
#         outputs = np.hstack([self.model(x, future=future)[:,-future:].cpu().detach().numpy() for i in range(num_samples)]) # n번 inference, output.shape = [20, N]
    # CONE ESTIMATION
    outputs = torch.zeros((x.shape[0], future, num_samples))
    
    for s in range(num_samples):
        new_gt = torch.Tensor([]).to(device)
        for f in range(future):
            if new_gt.size() != torch.Size([0]):
#                 x_new = torch.cat((x, new_gt.view(-1,1)),1)
                x_new = torch.cat((x_new, new_gt.view(-1,1)),1)
            else:
                x_new = x
#             new_gt = model(x, future=1)[:,-1]
            new_gt = model(x_new, future=1)[:,-1]
            outputs[:,f,s]=new_gt
    print('shape',outputs.shape)
#     print(outputs.mean(-1))
#     print(outputs.std(-1))
    y_variance = outputs.var(axis=-1)
    outputs = torch.Tensor(scaler.inverse_transform(outputs.cpu().detach().numpy()))
    tau = l2 * (1. - dropout_rate) / (2. * N * decay) # What is this doing??
    y_variance += (1. / tau)
    return outputs.mean(-1), outputs.std(-1)


# uncertainity_estimate(x, model, num_samples, l2, future)
# from dropout_rnn import DRNN, Optimization, generate_sequence, to_dataframe, inverse_transform, transform_data # Arka, feel free to change these names
FUTURE = 15
test_size = 100
N = x_train.shape[0]
print(N)
predicted_mean, predicted_sd = uncertainty_estimate(x_test[:test_size,:-FUTURE], model_1, num_samples=50, future=FUTURE, dropout_rate=dropout_frac, decay=decay, N=N)


