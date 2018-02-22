import torch
from torch.autograd import Variable
from torch import nn
from torch.utils.data import DataLoader
from torch import optim

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import dill  # dill extends python’s pickle module for serializing and de-serializing python objects
import shutil  # high level os functionality

import gc

from collections import defaultdict




class LoadFullDataset():
    def __init__(self, csv_path, train_valid_ratio=0.9, train_len=None, seq_length=96) -> None:
        self.dataset_values = pd.read_csv(csv_path).loc[:, 'actual'].values

        dataset_len = self.dataset_values.shape[0]

        # === CREATE PERIODIC SIGNALS
        daycount = self.dataset_values.shape[0] // seq_length
        self.dataset_values = self.dataset_values[:daycount*seq_length]  # remove uncomplete days

        def create_period_signal(freq, Fs):
            t = np.arange(Fs)
            return np.sin(2 * np.pi * freq * t / Fs)

        p_day = create_period_signal(daycount * seq_length / 96, daycount * seq_length)
        p_week = create_period_signal(daycount * seq_length / (96 * 7), daycount * seq_length)
        p_month = create_period_signal(daycount * seq_length / (96 * 30), daycount * seq_length)
        p_year = create_period_signal(daycount * seq_length / (96 * 365), daycount * seq_length)

        self.dataset_values = np.stack((self.dataset_values, p_day, p_week, p_month, p_year), axis=1)
        self.dataset_values = np.reshape(self.dataset_values, (-1, seq_length, 5))

        # SPLIT TRAIN & VALID
        if train_len is None:
            train_len = int(daycount * train_valid_ratio)
        valid_len = daycount - train_len


        train_values = self.dataset_values[:train_len, :, :]
        valid_values = self.dataset_values[train_len:, :, :]
        self.train_dataset = LoadDataset(train_values, seq_length=seq_length)
        self.valid_dataset = LoadDataset(valid_values, seq_length=seq_length)


class LoadDataset(torch.utils.data.Dataset):
    """

        Args:
            seq_length:
        Attributes:
            dataset:
            X:
            y:
    """

    def __init__(self, dataset, seq_length, shuffle=True):
        # normalize data. otherwise criterion cannot calculate loss
        dataset = (dataset - dataset.min()) / (dataset.max() - dataset.min())
        # split data wrt period
        # e.g. period = 96 -> (day_size, quarter_in_day)

        self.dataset = dataset

        if shuffle:
            np.random.shuffle(self.dataset)

        # rearrange X and targets
        # X = (d1,d2,d3...dn-1)
        # y = (d2,d3,d4...dn)
        self.y = self.dataset[1:, :, 0]
        self.X = self.dataset[:-1, :, :]

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

    def __init__(self, input_size, seq_length, num_layers, batch_size):
        super(LoadLSTM, self).__init__()
        self.input_size = input_size
        self.seq_length = seq_length
        self.num_layers = num_layers
        self.batch_size = batch_size

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


class Plotter:
    """
    Visualizes all related info.
    """

    def __init__(self, xlim, ylim=(0, 1), block=False):

        self.xlim = xlim
        self.ylim = ylim
        self.block = block

        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(1, 1, 1)

        plt.xlim(self.xlim)
        plt.ylim(self.ylim)
        plt.show(block=block)

        self.plot_container = defaultdict(tuple)

    def plot(self):
        """
        Plot all appended figures.

        Returns:

        """
        self.ax.clear()

        for (label, (plot_type, what_to_plot)) in self.plot_container.items():
            if plot_type == 'line':
                plt.plot(what_to_plot[-self.xlim[1]:], label=label)
            if plot_type == 'scatter':
                plt.scatter(x=list(range(self.xlim[1])), y=what_to_plot[-self.xlim[1]:], label=label)

        plt.legend()
        plt.pause(0.0001)
        self.fig.canvas.draw()

    def add(self, what_to_plot, plot_type, label):
        """
        Add what_to_plot to plot container.
        Args:
            what_to_plot:
            plot_type:
            label:

        Returns:

        """
        self.plot_container[label] = (plot_type, what_to_plot)

    def drop(self, label):
        """
        Drop label from plot container.
        Args:
            label:

        Returns:

        Raises: KeyError

        """
        self.plot_container.pop(label)


class History:
    """
        Args:
            what_to_store (list): list of labels for what to store.
    """

    def __init__(self, what_to_store):
        self.container = defaultdict(list)

        for w in what_to_store:
            self.container[w] = []

    def append(self, label, value):
        """

        Args:
            label (str):
            value (float):

        Returns:

        """
        if label not in self.container.keys():
            raise Exception('key:{} not available in history'.format(label))

        self.container[label].append(value)

    def get(self, label):
        """

        Args:
            label:

        Returns:
        Raises: KeyError
        """
        return self.container[label]

    def last(self, label):
        """

        Args:
            label:

        Returns:
        Raises: KeyError
        """
        arr = self.container[label]
        if len(arr) == 0:
            return np.inf

        return arr[-1]


class LoadEstimator:
    """
    todo: Please add docstring
    """

    # TODO: save experiment settings

    def __init__(self, config, resume=False):
        """

        Args:
            config:
        """
        self.config = config
        # if we seed random func, they will generate same output everytime.
        if config.RANDOM_SEED is not None:
            torch.manual_seed(config.RANDOM_SEED)
            np.random.seed(config.RANDOM_SEED)

        dataset = LoadFullDataset(csv_path=config.INPUT_PATH,
                                  train_valid_ratio=config.TRAIN_VALID_RATIO,
                                  train_len=config.TRAIN_LEN,
                                  seq_length=config.SEQ_LENGTH)

        self.train_dataset = dataset.train_dataset
        self.valid_dataset = dataset.valid_dataset

        self.train_dataloader = DataLoader(self.train_dataset, batch_size=config.BATCH_SIZE, drop_last=True)
        self.valid_dataloader = DataLoader(self.valid_dataset, batch_size=config.BATCH_SIZE, drop_last=True)

        self.model = LoadLSTM(input_size=config.INPUT_SIZE,
                                    seq_length=config.SEQ_LENGTH,
                                    num_layers=config.NUM_LAYERS,
                              batch_size=config.BATCH_SIZE)

        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adadelta(self.model.parameters(), lr=1.0)
        self.history = History(what_to_store=['train_loss', 'valid_loss', 'test_loss'])
        self.plotter = Plotter(xlim=(0, config.SEQ_LENGTH), ylim=(0, 1), block=False)

        self.experiment_dir = config.EXPERIMENT_DIR
        self.epoch = 0

        if resume:
            self.load_from_latest_ckpt()

    def load_from_latest_ckpt(self):
        """

        Returns:

        """
        latest_ckpt_path = Checkpoint.get_latest_checkpoint(self.experiment_dir)

        print("model reading from {} ...".format(latest_ckpt_path))

        latest_ckpt = Checkpoint.load(path=latest_ckpt_path)

        self.model = latest_ckpt.model
        self.optimizer = latest_ckpt.optimizer
        self.epoch = latest_ckpt.epoch + 1  # increment by 1 to train next epoch
        self.history = latest_ckpt.history

    def _train_on_batch(self, X_batch, y_batch):
        """

        Args:
            X_batch:
            y_batch:

        Returns:

        """

        self.optimizer.zero_grad()  # pytorch accumulates gradients.
        gc.collect()
        self.model.hidden = self.model.init_hidden()  # detach history of initial hidden
        lstm_out, hidden = self.model(X_batch)

        prediction = hidden[0][-1, :, :]
        loss = self.criterion(prediction, y_batch)

        loss.backward()
        self.optimizer.step()

        return lstm_out, hidden, prediction, loss

    def _train_on_epoch(self, epoch):
        """

        Args:
            epoch:

        Returns:

        """
        self.model.train(mode=True)

        for batch_num, (X, y) in enumerate(self.train_dataloader):
            batch_size = X.size()[1]
            step = batch_size * batch_num

            (X, y) = Variable(X.float(), requires_grad=False), Variable(y.float(), requires_grad=False)
            (lstm_out, hidden, prediction, train_loss) = self._train_on_batch(X_batch=X, y_batch=y)

            self.history.append(label='train_loss', value=train_loss.data.numpy()[0].item())

            print("epoch : {:>8} || batch_num : {:>8} || train_loss : {:.5f} || valid_loss  {:.5f}".format(
                epoch, batch_num, self.history.last('train_loss'), self.history.last('valid_loss')))

            if (batch_num + 1) % 10 == 0:
                X_to_plot = X.data.numpy()[:, :, 0].flatten()
                prediction_to_plot = prediction.data.numpy().flatten()
                self.plotter.add(what_to_plot=X_to_plot, plot_type='line', label='true')
                self.plotter.add(what_to_plot=prediction_to_plot, plot_type='line', label='pred')
                self.plotter.add(what_to_plot=self.history.get('train_loss'), plot_type='line', label='train_loss')
                self.plotter.add(what_to_plot=self.history.get('valid_loss'), plot_type='line', label='valid_loss')
                self.plotter.plot()

        # save model, optimizer, epoch, history to the experiment_dir/datetime_epoch
        Checkpoint(model=self.model, optimizer=self.optimizer,
                   epoch=self.epoch, history=self.history,
                   experiment_dir=self.experiment_dir).save()

    def _validate(self):
        # TODO: Implement self._validate and append loss to the history container
        self.model.eval()
        valid_losses = []
        for batch_num, (X, y) in enumerate(self.valid_dataloader):

            (X, y) = Variable(X.float(), requires_grad=False), Variable(y.float(), requires_grad=False)
            self.model.hidden = self.model.init_hidden()
            lstm_out, hidden  = self.model(X)

            prediction = Variable(hidden[0][-1, :, :].data, requires_grad=False)
            valid_loss = self.criterion(prediction, y)

            valid_losses.append(valid_loss.data.numpy()[0].item())

        self.history.append(label='valid_loss', value=sum(valid_losses)/len(valid_losses))


    def _test(self, X, y):
        # TODO: Implement self._test and append loss to the history container
        # self.history.append(label='test_loss', value=test_loss.data.numpy()[0])
        pass

    def train(self, epoch_size=20):
        """
        Iterates from latest epoch to epoch_size because maybe model is resuming from latest checkpoint.
        Updates self.epoch every time too to be ready for next saving process.
        Args:
            epoch_size: how many epoch do we want to train the model?

        Returns:

        """
        for self.epoch in range(self.epoch, epoch_size):
            self._train_on_epoch(epoch=self.epoch)
            self._validate()
        # self._test()


class Config:
    """

    """

    def __init__(self):
        """

        """
        self.SEQ_LENGTH = 96
        self.NUM_LAYERS = 1
        self.BATCH_SIZE = 20
        self.INPUT_SIZE = 5
        self.INPUT_PATH = '../input/load_wo_feb29.csv'
        self.EXPERIMENT_DIR = '../experiments/experiment_real_1layer_wo_feb29'
        self.RANDOM_SEED = 7
        self.TRAIN_VALID_RATIO = 0.95
        self.TRAIN_LEN = 2600  # 2600 days * 96 quarter out of 2922 days

        self.RESUME = False


class Checkpoint:
    """
    Args:

        model (LoadLSTM): loadmodel
        optimizer (optim): stores the state of the optimizer
        epoch (int): current epoch (an epoch is a loop through the full training data)
        step (int): number of examples seen within the current epoch
        # input (np.array):
        # output (np.array):

    Attributes:
        CHECKPOINT_DIR_NAME (str):
        TRAINER_STATE_NAME (str):
        MODEL_NAME (str):
        # INPUT_FILE (str):
        # OUTPUT_FILE (str):

    """
    CHECKPOINT_DIR_NAME = 'checkpoints'
    TRAINER_STATE_NAME = 'trainer_states.pt'
    MODEL_NAME = 'model.pt'

    # INPUT_FILE = 'input.pt'
    # OUTPUT_FILE = 'output.pt'

    def __init__(self, model, optimizer, epoch, history, experiment_dir):
        """

        Args:
            model:
            optimizer:
            epoch:
            path:
        """
        self.model = model
        self.optimizer = optimizer
        self.epoch = epoch
        self.history = history
        self.experiment_dir = experiment_dir

    def save(self):
        """

        Args:
            experiment_dir:

        Returns:

        """
        date_time = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime())
        save_path = '{}_epoch_{}'.format(date_time, self.epoch)
        subdir_path = os.path.join(self.experiment_dir, self.CHECKPOINT_DIR_NAME, save_path)

        if os.path.exists(subdir_path):
            # Clear dir
            # os.removedirs(subdir_path) fails if subdir_path is not empty.
            shutil.rmtree(subdir_path)

        os.makedirs(subdir_path)

        # SAVE
        torch.save({'epoch': self.epoch,
                    'optimizer': self.optimizer,
                    'history': self.history},
                   os.path.join(subdir_path, self.TRAINER_STATE_NAME))
        torch.save(self.model, os.path.join(subdir_path, self.MODEL_NAME))

        # with open(os.path.join(subdir_path, self.INPUT_FILE), 'wb') as fout:
        #     dill.dump(self.input, fout)
        # with open(os.path.join(subdir_path, self.OUTPUT_FILE), 'wb') as fout:
        #     dill.dump(self.output, fout)

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

        with open(os.path.join(path, cls.INPUT_FILE), 'rb') as fin:
            input = dill.load(fin)

        with open(os.path.join(path, cls.OUTPUT_FILE), 'rb') as fin:
            output = dill.load(fin)

        return Checkpoint(model=model,
                          optimizer=resume_checkpoint['optimizer'],
                          epoch=resume_checkpoint['epoch'],
                          history=resume_checkpoint['history'],
                          experiment_dir=path)

    @classmethod
    def get_latest_checkpoint(cls, experiment_path):
        """

        Precondition: Assumes experiment_path exists and have at least 1 checkpoint

        Args:
            experiment_path:

        Returns:

        """
        checkpoints_path = os.path.join(experiment_path, cls.CHECKPOINT_DIR_NAME)
        all_times = sorted(os.listdir(checkpoints_path), reverse=True)
        return os.path.join(checkpoints_path, all_times[0])



config = Config()

estimator = LoadEstimator(config=config, resume=config.RESUME)
estimator.train(epoch_size=40)
