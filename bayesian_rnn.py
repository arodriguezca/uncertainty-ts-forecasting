'''
    Adapted to time series from https://github.com/JACKHAHA363/BBBRNN/blob/master/main.py
    by Alex
'''

import torch.nn as nn
from BBBLayers import BBBLinear
from BBBLayers import BBBRNN
from torch.autograd import Variable
import torch
import numpy as np
import random

# class BBBRNNModel(nn.Module):
#     """
#     Modify from language model pytorch exampl
#     """
#     def __init__(
#             self, rnn_type, sharpen, ntoken, ninp,
#             nhid, nlayers, dropout=0.5,
#             tie_weights=False, *args, **kwargs
#     ):
#         super(BBBRNNModel, self).__init__()
#         self.drop = nn.Dropout(dropout)
#         self.encoder = nn.Embedding(ntoken, ninp)
#         self.sharpen = sharpen
#         self.rnn = BBBRNN(
#             rnn_type, sharpen, ninp, nhid, nlayers, dropout=dropout,
#             *args, **kwargs
#         )

#         self.decoder = BBBLinear(nhid, ntoken, *args, **kwargs)

#         if tie_weights:
#             if nhid != ninp:
#                 raise ValueError('When using the tied flag, nhid must be equal to emsize')
#             self.decoder.weight_mean = self.encoder.weight

#         # init embedding
#         initrange = 0.1
#         self.encoder.weight.data.uniform_(-initrange, initrange)

#         self.rnn_type = rnn_type
#         self.ninp = ninp
#         self.nhid = nhid
#         self.nlayers = nlayers
#         self.ntoken = ntoken

#         self.layers = [self.rnn, self.decoder]
        


#     def forward(self, input, hidden, targets):
#         """
#         :param input: [seq_len, bsz, inp_dim]
#         :return: [seq_len, bsz, inp_dim]
#         """
#         emb = self.drop(self.encoder(input))
#         output, hidden = self.rnn(emb, hidden)
#         output = self.drop(output)
#         decoded = self.decoder(
#             output.view(output.size(0)*output.size(1), output.size(2))
#         )
#         outputs = decoded.view(output.size(0), output.size(1), decoded.size(1))
#         if self.sharpen and self.training:
#             # We compute the cost
#             NLL = self.get_nll(outputs, targets)
#             # The gradients
#             gradients = torch.autograd.grad(outputs=NLL, inputs=self.rnn.sampled_weights, grad_outputs=torch.ones(NLL.size()).cuda(), create_graph=True, retain_graph=True, only_inputs=True)
#             # Then we do the forward pass again with sharpening:
#             output, hidden = self.rnn(emb, hidden, gradients)
#             decoded = self.decoder(
#                     output.view(output.size(0)*output.size(1), output.size(2))
#                     )
#             outputs = decoded.view(output.size(0), output.size(1), decoded.size(1))
#         return outputs, hidden


    # def init_hidden(self, bsz):
    #     weight = next(self.parameters()).data
    #     if self.rnn_type == 'LSTM':
    #         return (Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()),
    #                 Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()))
    #     else:
    #         return Variable(weight.new(self.nlayers, bsz, self.nhid).zero_())


    
# def repackage_hidden(h):
#     """Wraps hidden states in new Variables, to detach them from their history."""
#     if type(h) == Variable:
#         return Variable(h.data)
#     else:
#         return tuple(repackage_hidden(v) for v in h)



class Optimization_BBBRNN:
    """ A helper class to train, test and diagnose the LSTM"""

    def __init__(self, model, loss_fn, optimizer, scheduler):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_losses = []
        self.val_losses = []
        self.futures = []
        self.sharpen = False
        self.loss_fn = loss_fn
        self.bptt = 35
        self.rnn = self.model.lstm
        self.clip = 5 # gradient clipping
        self.layers = [self.rnn, self.model.linear]

    @staticmethod
    def generate_batch_data(x, y, batch_size):
        for batch, i in enumerate(range(0, len(x) - batch_size, batch_size)):
            x_batch = x[i : i + batch_size]
            y_batch = y[i : i + batch_size]
            yield x_batch, y_batch, batch

    def get_nll(self, output, targets):
        # \sum log P(batch | theta) / (bsz * seq_len)
        return self.loss_fn(output, targets)  # NOTE: it should be view(-1, num_tokens)

    def get_loss(self, output, targets):
        """
        return:
            NLL: NLL is averaged over seq_len and batch_size
            KL: KL is the original scale KL
        """
        # NLL
        NLL = self.get_nll(output, targets)

        # KL
        KL = torch.zeros(1)
        if self.rnn.gpu:
            KL = KL.cuda()
        KL = Variable(KL)

        for layer in self.layers:
            if isinstance(layer, BBBRNN):
                KL += layer.get_kl()

        if self.sharpen:
            KL_sharp = self.rnn.get_kl_sharpening()
        else:
            KL_sharp = 0.
        return NLL, KL, KL_sharp

    def init_hidden(self, bsz):
        weight = next(self.model.parameters()).data
        if self.rnn.mode == 'LSTM':
            return (Variable(weight.new(self.rnn.num_layers, bsz, self.rnn.hidden_size).zero_()),
                    Variable(weight.new(self.rnn.num_layers, bsz, self.rnn.hidden_size).zero_()))
        else:
            return Variable(weight.new(self.rnn.num_layers, bsz, self.rnn.hidden_size).zero_())

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
        
        self.model.train()
        import time
        start_time = time.time()
        # ntokens = len(corpus.dictionary)
        hidden = self.init_hidden(batch_size)
        seq_len = x_train.shape[1]
        num_batch = x_train.size(0) / seq_len # args.bptt
        
        for epoch in range(n_epochs):
            start_time = time.time()
            self.futures = []

            train_loss = 0
            for x_batch, y_batch, batch in self.generate_batch_data(x_train, y_train, batch_size):
                y_pred = self._predict(x_batch, y_batch, seq_len, do_teacher_forcing)
                self.optimizer.zero_grad()
                # print(y_pred.shape)
                # print('batch',y_batch.shape)
                # print(y_batch)
                NLL, KL, KL_sharp = self.get_loss(y_pred, y_batch)
                # import sys
                # sys.exit()
                # proper scaling for a batch loss
                NLL_term = NLL * seq_len # self.bptt # \frac{1}{C} \sum_{c=1}^C p(y^c|x^c)
                KL_term = KL / (num_batch * batch_size) # KL(q|p) / BC
                loss = NLL_term + KL_term
                if self.sharpen:
                    loss += KL_sharp / num_batch
                loss.backward()

                # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
                # torch.nn.utils.clip_grad_norm(self.model.parameters(), self.clip)
                # for p in self.model.parameters():  NOTE: Alex: what is this for?
                #     p.data.add_(-lr, p.grad.data)

                # train_loss += loss.data
                # total_nll += NLL.data
                # total_kl += KL.data
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
                y_pred = self.model(x_batch, future=future)
                y_pred = (
                    y_pred[:, -len(y_batch) :] if y_pred.shape[1] > y_batch.shape[1] else y_pred
                )
                loss = self.loss_fn(y_pred, y_batch)
                test_loss += loss.item()
                actual += torch.squeeze(y_batch[:, -1]).data.cpu().numpy().tolist()
                predicted += torch.squeeze(y_pred[:, -1]).data.cpu().numpy().tolist()
            test_loss /= batch 
            return actual, predicted, test_loss

    # def plot_losses(self):
    #     plt.plot(self.train_losses, label="Training loss")
    #     plt.plot(self.val_losses, label="Validation loss")
    #     plt.legend()
    #     plt.title("Losses")