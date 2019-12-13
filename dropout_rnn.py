import torch.nn as nn
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt
import time
import random
import numpy as np
from torch.autograd import Variable
import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')


class DRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout_frac, rnn=nn.LSTMCell):
        '''
            @param drnn: rnn model of choice with dropout - insert here ours
        '''
        super(DRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.lstm = rnn(self.input_size, self.hidden_size)
        self.linear = nn.Linear(self.hidden_size, self.output_size)
        self.dropout_frac = dropout_frac
        self.dropout = nn.Dropout(p=self.dropout_frac)

    def forward(self, input, future=0, y=None):
        outputs = []

        # reset the state of LSTM
        # the state is kept till the end of the sequence
        h_t = torch.zeros(input.size(0), self.hidden_size, dtype=torch.float32).to(device)
        c_t = torch.zeros(input.size(0), self.hidden_size, dtype=torch.float32).to(device)
        
#         if dropout!=0:
#             input = dropout(input)
        
        for i, input_t in enumerate(input.chunk(input.size(1), dim=1)):
            h_t, c_t = self.lstm(input_t.to(device), (h_t, c_t))
            if self.dropout_frac != 0:
                h_t = self.dropout(h_t)
            output = self.linear(h_t)
            outputs += [output]

        for i in range(future):
            if y is not None and random.random() > 0.5:
                output = y[:, [i]]  # teacher forcing
            h_t, c_t = self.lstm(output, (h_t, c_t))
            output = self.linear(h_t)
            outputs += [output]
        outputs = torch.stack(outputs, 1).squeeze(2)
        return outputs


class Optimization:
    """ A helper class to train, test and diagnose the LSTM"""

    def __init__(self, model, loss_fn, optimizer, scheduler):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_losses = []
        self.val_losses = []
        self.futures = []

    @staticmethod
    def generate_batch_data(x, y, batch_size):
        for batch, i in enumerate(range(0, len(x) - batch_size, batch_size)):
            x_batch = x[i : i + batch_size]
            y_batch = y[i : i + batch_size]
            yield x_batch, y_batch, batch

    def train(
        self,
        x_train,
        y_train,
        x_val=None,
        y_val=None,
        batch_size=32,
        n_epochs=15,
        do_teacher_forcing=None,
    ):
        seq_len = x_train.shape[1]
        for epoch in range(n_epochs):
            start_time = time.time()
            self.futures = []

            train_loss = 0
            for x_batch, y_batch, batch in self.generate_batch_data(x_train, y_train, batch_size):
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                y_pred = self._predict(x_batch, y_batch, seq_len, do_teacher_forcing)
                self.optimizer.zero_grad()
                loss = self.loss_fn(y_pred, y_batch)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()
            self.scheduler.step()
            train_loss /= batch
            self.train_losses.append(train_loss)

            self._validation(x_val, y_val, batch_size)

            elapsed = time.time() - start_time
            print(
                "Epoch %d Train loss: %.2f. Validation loss: %.2f. Avg future: %.2f. Elapsed time: %.2fs."
                % (epoch + 1, train_loss, self.val_losses[-1], np.average(self.futures), elapsed)
            )

    def _predict(self, x_batch, y_batch, seq_len, do_teacher_forcing):
        if do_teacher_forcing:
            future = random.randint(1, int(seq_len) / 2)
            limit = x_batch.size(1) - future
            y_pred = self.model(x_batch[:, :limit], future=future, y=y_batch[:, limit:])
        else:
            future = 0
            y_pred = self.model(x_batch)
        self.futures.append(future)
        return y_pred

    def _validation(self, x_val, y_val, batch_size):
        if x_val is None or y_val is None:
            return
        with torch.no_grad():
            val_loss = 0
            batch = 1
            for x_batch, y_batch, batch in self.generate_batch_data(x_val, y_val, batch_size):
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                y_pred = self.model(x_batch)
                loss = self.loss_fn(y_pred, y_batch)
                val_loss += loss.item()
            val_loss /= batch
            self.val_losses.append(val_loss)

    def evaluate(self, x_test, y_test, batch_size, future=1):
        with torch.no_grad():
            test_loss = 0
            actual, predicted = [], []
            for x_batch, y_batch, batch in self.generate_batch_data(x_test, y_test, batch_size):
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                y_pred = self.model(x_batch, future=future)
                y_pred = (
                    y_pred[:, -len(y_batch) :] if y_pred.shape[1] > y_batch.shape[1] else y_pred
                )
                loss = self.loss_fn(y_pred.to(device), y_batch.to(device))
                test_loss += loss.item()
                actual += torch.squeeze(y_batch[:, -1]).data.cpu().numpy().tolist()
                predicted += torch.squeeze(y_pred[:, -1]).data.cpu().numpy().tolist()
            test_loss /= batch 
            return actual, predicted, test_loss
    
    def evaluate_all(self, x_test, y_test, batch_size, future=1):
        l2 = 0.01
        with torch.no_grad():
            test_loss = 0
            predicted_mean, predicted_sd = np.array([]), np.array([])
            
            for x_batch, y_batch, batch in self.generate_batch_data(x_test, y_test, batch_size):
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                y_pred_mean, y_pred_std = self.uncertainty_estimate(x_batch, self.model, 100, l2, future)
#                 self.model(x_batch, future=future)
#                 y_pred = (
#                     y_pred_mean[:, -len(y_batch) :] if y_pred_mean.shape[1] > y_batch.shape[1] else y_pred_mean
#                 )
#                 loss = self.loss_fn(y_pred, y_batch)
#                 test_loss += loss.item()
#                 print(y_batch.shape)
#                 print(y_pred_mean.shape)
#                 print(y_pred_std.shape)
#                 actual += torch.squeeze(y_batch[:, -1]).data.cpu().numpy().tolist()
                print('>>',predicted_mean.shape, y_pred_mean.shape)
                
                if predicted_mean.size == 0:
                    predicted_mean = y_pred_mean
                    predicted_sd = y_pred_std
                else:
                    predicted_mean = np.concatenate((predicted_mean, y_pred_mean),0)
                    predicted_sd = np.concatenate((predicted_sd, y_pred_std),0)
                
#             test_loss /= batch 
            return predicted_mean, predicted_sd
        
    def uncertainty_estimate(self, x, model, num_samples, l2, future):
        x = x.to(device)
#         print(x.shape)
#         print(self.model(x, future=future)[:,-future:].cpu().detach().numpy())
        print(self.model(x, future=future)[:,-future:].cpu().detach().numpy().shape)
#         outputs = np.hstack([self.model(x, future=future)[:,-future:].cpu().detach().numpy() for i in range(num_samples)]) # n번 inference, output.shape = [20, N]
        
    
        def predict_given_gt_sequence(x, model, num_samples):
            '''
                num_samples: number of samples per ground truth sequence
            '''
            outputs = np.zeros((x.shape[0], num_samples))
            for i in range(num_samples):
                output_sample = self.model(x, future=1)[:,-1].cpu().detach().numpy()
                outputs[:,i] = output_sample
            return outputs
                   
        # CONE ESTIMATION
        for i in range(future):
            if i == 0:
                outputs = predict_given_gt_sequence(x, model, num_samples)
                y_mean = outputs.mean(axis=-1)
                y_var = outputs.var(axis=-1)
            else:
                all_outputs = np.array([])
                for _ in range(30):
                    # update x
                    y_std = y_var ** 0.5
                    new_gt = torch.normal(y_mean, y_std)
                    x_new = torch.cat((x.copy(), new_gt),1)
                    print('x_new', x_new.shape)
                    outputs = predict_given_gt_sequence(x_new, model, num_samples)
                    if all_outputs.size == 0:
                        all_outputs = outputs
                    else:
                        all_outputs = np.concatenate((all_outputs, outputs),1)
                        print(all_outputs.shape)

        # PREVIOUS:
#         outputs = np.zeros((x.shape[0], future, num_samples))
#         for i in range(num_samples):
#             output_sample = self.model(x, future=future)[:,-future:].cpu().detach().numpy()
#             outputs[:,:,i]=output_sample
#         print('o',outputs.shape)
#         y_mean = outputs.mean(axis=-1)
#         y_variance = outputs.var(axis=-1)
#         tau = l2 * (1. - model.dropout_rate) / (2. * N * model.decay) # What is this doing??
#         y_variance += (1. / tau)
        y_std = np.sqrt(y_variance)
        return y_mean, y_std
    
    def plot_losses(self):
        plt.plot(self.train_losses, label="Training loss")
        plt.plot(self.val_losses, label="Validation loss")
        plt.legend()
        plt.title("Losses")


def transform_data(arr, seq_len):
    x, y = [], []
    for i in range(len(arr) - seq_len):
        x_i = arr[i : i + seq_len]
        y_i = arr[i + 1 : i + seq_len + 1]
        x.append(x_i)
        y.append(y_i)
    x_arr = np.array(x).reshape(-1, seq_len)
    y_arr = np.array(y).reshape(-1, seq_len)
    x_var = Variable(torch.from_numpy(x_arr).float())
    y_var = Variable(torch.from_numpy(y_arr).float())
    return x_var, y_var

def generate_sequence(scaler, model, x_sample, future=1000):
    """ Generate future values for x_sample with the model """
    y_pred_tensor = model(x_sample, future=future)
    y_pred = y_pred_tensor.cpu().tolist()
    y_pred = scaler.inverse_transform(y_pred)
    return y_pred

def to_dataframe(actual, predicted):
    return pd.DataFrame({"actual": actual, "predicted": predicted})

def inverse_transform(scaler, df, columns):
    for col in columns:
        df[col] = scaler.inverse_transform(df[col])
    return df