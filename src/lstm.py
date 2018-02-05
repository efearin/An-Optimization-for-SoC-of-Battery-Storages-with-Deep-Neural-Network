import torch
from torch.autograd import Variable
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch import optim

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import dill  # dill extends python’s pickle module for serializing and de-serializing python objects
import shutil  # high level os functionality


class LoadDataset(Dataset):
    """
        Reads and normalizes the data from given csv_file.
        Args:
            csv_path:
            seq_length:
        Attributes:
            dataset:
            X:
            y:
    """

    def __init__(self, csv_path, seq_length=96):
        """

        Args:
            csv_path:
            seq_length:
        """
        self.dataset = pd.read_csv(csv_path).loc[:, 'actual'].values

        # normalize data. otherwise criterion cannot calculate loss
        self.dataset = (self.dataset - self.dataset.min()) / (self.dataset.max() - self.dataset.min())
        # split data wrt period
        # e.g. period = 96 -> (day_size, quarter_in_day)
        datacount = self.dataset.shape[0] // seq_length

        self.X = self.dataset[:(datacount * seq_length)]

        def create_period_signal(freq, Fs):
            t = np.arange(Fs)
            return np.sin(2 * np.pi * freq * t / Fs)

        p_day = create_period_signal(datacount * seq_length / 96, datacount * seq_length)
        p_week = create_period_signal(datacount * seq_length / (96 * 7), datacount * seq_length)
        p_month = create_period_signal(datacount * seq_length / (96 * 30), datacount * seq_length)
        p_year = create_period_signal(datacount * seq_length / (96 * 365), datacount * seq_length)

        self.X = np.stack((self.X, p_day, p_week, p_month, p_year), axis=1)

        self.X = np.reshape(self.X, (-1, seq_length, 5))
        # rearrange X and targets
        # X = (d1,d2,d3...dn-1)
        # y = (d2,d3,d4...dn)
        self.y = self.X[1:, :, 0]
        self.X = self.X[:-1, :, :]

    def __len__(self):
        """

        Returns:
            int: data count

        """
        return self.X.shape[0]

    def __getitem__(self, ix):
        """

        Args:
            ix:

        Returns:
            (np.ndarray, np.ndarray):

        """
        # (row, seq_len, input_size)
        return self.X[ix, :, :], self.y[ix, :]


class LoadLSTM(nn.Module):
    """
        Long-short term memory implementation for LoadDataset

        Args:
            input_size:
            seq_length:
            num_layers:

        Attributes:
            input_size:
            seq_length:
            num_layers:
            lstm:
            hidden:
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
        """

        Returns:
            (Variable,Variable): (h_0, c_0)

        """
        # (num_layers, batch, hidden_dim)
        return (Variable(torch.zeros(self.num_layers, 1, self.seq_length)),  # h_0
                Variable(torch.zeros(self.num_layers, 1, self.seq_length)))  # c_0

    def forward(self, x):
        """

        Args:
            x:

        Returns:
            (?,?)
        """
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
        """

        Args:
            config:
        """
        self.config = config
        self.dataset = LoadDataset(csv_path='../input/load.csv', seq_length=config.SEQ_LENGTH)
        self.dataloader = DataLoader(self.dataset, batch_size=config.BATCH_SIZE)
        self.model = LoadLSTM(input_size=config.INPUT_SIZE, seq_length=config.SEQ_LENGTH, num_layers=config.NUM_LAYERS)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adadelta(self.model.parameters(), lr=1.0)

    def train(self, epoch_size=20):
        """

        Args:
            epoch_size:

        Returns:

        """
        for epoch in range(epoch_size):
            for i, (X_batch, y_batch) in enumerate(self.dataloader):
                (X_batch, y_batch) = Variable(X_batch.float()), Variable(y_batch.float())

                self.optimizer.zero_grad()  # pytorch accumulates gradients.
                self.model.hidden = self.model.init_hidden()  # detach history of initial hidden
                lstm_out, hidden = self.model(X_batch)  # TODO: continue from here...

                pred = hidden[0][-1, :, :]
                loss = self.criterion(pred, y_batch)

                loss.backward()
                self.optimizer.step()

                self.plot(epoch, loss, X_batch, pred)

    def plot(self, epoch, loss, X_batch, pred):
        """

        Args:
            epoch:
            loss:
            X_batch:
            pred:

        Returns:

        """
        print("epoch : {} || loss : {}".format(epoch, loss.data.numpy()))
        ax.clear()
        plt.plot(X_batch.data.numpy()[:, :, 0].flatten().reshape(
            self.config.SEQ_LENGTH * self.config.BATCH_SIZE), label='true')
        plt.plot(pred.data.numpy().flatten().reshape(
            self.config.SEQ_LENGTH * self.config.BATCH_SIZE), label='pred')
        plt.legend()
        plt.xlim([0, self.config.SEQ_LENGTH])
        plt.ylim([0, 1])
        plt.pause(0.0001)
        fig.canvas.draw()


class Config:
    """

    """
    def __init__(self):
        """

        """
        self.SEQ_LENGTH = 96
        self.NUM_LAYERS = 6
        self.BATCH_SIZE = 1
        self.INPUT_SIZE = 5


class Checkpoint:
    """
    Args:
        model (LoadLSTM):
        optimizer (optim):
        epoch (int):
        step (int):
        input (np.array):
        output (np.array):

    Attributes:
        CHECKPOINT_DIR_NAME (str):
        TRAINER_STATE_NAME (str):
        MODEL_NAME (str):
        INPUT_FILE (str):
        OUTPUT_FILE (str):

    """
    CHECKPOINT_DIR_NAME = 'checkpoints'
    TRAINER_STATE_NAME = 'trainer_states.pt'
    MODEL_NAME = 'model.pt'
    INPUT_FILE = 'input.pt'
    OUTPUT_FILE = 'output.pt'

    def __init__(self, model, optimizer, epoch, step, input, output, path):
        """

        Args:
            model:
            optimizer:
            epoch:
            step:
            input:
            output:
            path:
        """
        self.model = model
        self.optimizer = optimizer
        self.epoch = epoch
        self.step = step
        self.input = input
        self.output = output
        self.path = path

    def save(self, experiment_dir):
        """

        Args:
            experiment_dir:

        Returns:

        """
        date_time = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime())
        subdir_path = os.path.join(experiment_dir, self.CHECKPOINT_DIR_NAME, date_time)

        if os.path.exists(subdir_path):
            # Clear dir
            # os.removedirs(subdir_path) fails if subdir_path is not empty.
            shutil.rmtree(subdir_path)

        os.makedirs(subdir_path)

        # SAVE
        torch.save({'epoch': self.epoch,
                    'step': self.step,
                    'optimizer': self.optimizer},
                   os.path.join(subdir_path, self.TRAINER_STATE_NAME))
        torch.save(self.model, os.path.join(subdir_path, self.MODEL_NAME))

        with open(os.path.join(subdir_path, self.INPUT_FILE), 'wb') as fout:
            dill.dump(self.input, fout)
        with open(os.path.join(subdir_path, self.OUTPUT_FILE), 'wb') as fout:
            dill.dump(self.output, fout)

    @classmethod
    def load(cls, path):
        """

        Args:
            path (str):

        Returns:
            Checkpoint:

        """
        resume_checkpoint = torch.load(os.path.join(path, cls.TRAINER_STATE_NAME))
        model = torch.load(os.path.join(path, cls.MODEL_NAME))

        model.flatten_parameters()  # make RNN parameters contiguos
        with open(os.path.join(path, cls.INPUT_FILE), 'rb') as fin:
            input = dill.load(fin)

        with open(os.path.join(path, cls.OUTPUT_FILE), 'rb') as fin:
            output = dill.load(fin)

        return Checkpoint(model=model,
                          optimizer=resume_checkpoint['optimizer'],
                          epoch=resume_checkpoint['epoch'],
                          step=resume_checkpoint['step'],
                          input=input,
                          output=output,
                          path=path)

    @classmethod
    def get_latest_checkpoint(cls, experiment_path):
        """

        Args:
            experiment_path:

        Returns:

        """
        checkpoints_path = os.path.join(experiment_path, cls.CHECKPOINT_DIR_NAME)
        all_times = sorted(os.listdir(checkpoints_path), reverse=True)
        return os.path.join(checkpoints_path, all_times[0])


if __name__ == "__main__":
    config = Config()
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    plt.show(block=False)

    estimator = LoadEstimator(config=config)
    estimator.train(epoch_size=20)

    Checkpoint.get_latest_checkpoint('lel')
