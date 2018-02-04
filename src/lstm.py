import torch
from torch.autograd import Variable
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch import optim

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time


class LoadDataset(Dataset):
    """
    todo: insert docstring
    """
    def __init__(self, csv_path, seq_length=96):
        """"""
        self.dataset = pd.read_csv(csv_path).loc[:, 'actual'].values

        # normalize data. otherwise criterion cannot calculate loss
        self.dataset = (self.dataset - self.dataset.min()) / (self.dataset.max() - self.dataset.min())
        # split data wrt period
        # e.g. period = 96 -> (day_size, quarter_in_day)
        datacount = self.dataset.shape[0] // seq_length

        self.X = self.dataset[:(datacount*seq_length)]

        def create_period_signal(freq, Fs):
            t = np.arange(Fs)
            return np.sin(2 * np.pi * freq * t / Fs)

        p_day = create_period_signal(datacount*seq_length/96, datacount*seq_length)
        p_week = create_period_signal(datacount*seq_length/(96*7), datacount*seq_length)
        p_month = create_period_signal(datacount*seq_length/(96*30), datacount*seq_length)
        p_year = create_period_signal(datacount*seq_length/(96*365), datacount*seq_length)

        self.X = np.stack((self.X, p_day, p_week, p_month, p_year), axis=1)


        self.X = np.reshape(self.X, (-1, seq_length, 5))
        # rearrange X and targets
        # X = (d1,d2,d3...dn-1)
        # y = (d2,d3,d4...dn)
        self.y = self.X[1:,:, 0]
        self.X = self.X[:-1,:, :]


    def __len__(self):
        """"""
        return self.X.shape[0]

    def __getitem__(self, ix):
        """"""
        # (row, seq_len, input_size)
        return self.X[ix, :, :], self.y[ix, :]

class LoadLSTM(nn.Module):
    """
    todo: insert docstring
    """

    def __init__(self, input_size, seq_length, num_layers):
        super(LoadLSTM, self).__init__()
        self.input_size = input_size
        self.seq_length = seq_length
        self.num_layers = num_layers

        # Inputs: input, (h_0,c_0)
        #   input(seq_len, batch, input_size)
        #   h_0(num_layers * num_directions, batch, hidden_size)
        #   c_0(num_layers * num_directions, batch, hidden_size)
        #   Note:If (h_0, c_0) is not provided, both h_0 and c_0 default to zero.
        # Outputs: output, (h_n, c_n)
        #   output (seq_len, batch, hidden_size * num_directions)
        #   h_n (num_layers * num_directions, batch, hidden_size)
        #   c_n (num_layers * num_directions, batch, hidden_size)
        self.lstm = nn.LSTM(input_size=self.input_size,
                            hidden_size=self.seq_length,
                            num_layers=self.num_layers)

        self.hidden = self.init_hidden()

    def init_hidden(self):
        # (num_layers, batch, hidden_dim)
        return (Variable(torch.zeros(self.num_layers, 1, self.seq_length)),  # h_0
                Variable(torch.zeros(self.num_layers, 1, self.seq_length)))  # c_0

    def forward(self, x):
        # Reshape input
        # x shape: (seq_len, batch, input_size)
        # hidden shape:(num_layers * num_directions,batch_size, hidden_size)
        x = x.view(self.seq_length, -1, self.input_size)
        lstm_out, self.hidden = self.lstm(x, self.hidden)
        return lstm_out, self.hidden

class LoadEstimator:
    """
    todo: Please add docstring
    """
    def __init__(self, config):
        self.config = config
        self.dataset = LoadDataset(csv_path='../input/load.csv', seq_length=config.SEQ_LENGTH)
        self.dataloader = DataLoader(self.dataset, batch_size=config.BATCH_SIZE)
        self.model = LoadLSTM(input_size=config.INPUT_SIZE, seq_length=config.SEQ_LENGTH, num_layers=config.NUM_LAYERS)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adadelta(self.model.parameters(), lr=1.0)

    def train(self, epoch_size=20):
        for epoch in range(epoch_size):
            for i, (X_batch, y_batch) in enumerate(self.dataloader):
                (X_batch, y_batch) = Variable(X_batch.float()), Variable(y_batch.float())

                self.optimizer.zero_grad()
                self.model.hidden = self.model.init_hidden()
                lstm_out, hidden = self.model(X_batch)

                pred = hidden[0][-1,:,:]
                loss = self.criterion(pred, y_batch)

                loss.backward()
                self.optimizer.step()

                self.plot(epoch, loss, X_batch, pred)

    def plot(self, epoch, loss, X_batch, pred):
        # print("epoch : {} || loss : {}".format(epoch, loss.data.numpy()))
        ax.clear()
        plt.plot(X_batch.data.numpy()[:,:,0].flatten().reshape(
            self.config.SEQ_LENGTH*self.config.BATCH_SIZE), label='true')
        plt.plot(pred.data.numpy().flatten().reshape(
            self.config.SEQ_LENGTH*self.config.BATCH_SIZE), label='pred')
        plt.legend()
        plt.xlim([0,self.config.SEQ_LENGTH])
        plt.ylim([0, 1])
        plt.pause(0.0001)
        fig.canvas.draw()

class Config:
    def __init__(self):
        self.SEQ_LENGTH = 96
        self.NUM_LAYERS = 6
        self.BATCH_SIZE = 1
        self.INPUT_SIZE = 5


if __name__ == "__main__":

    config = Config()
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    plt.show(block=False)

    estimator = LoadEstimator(config=config)
    estimator.train(epoch_size=20)






