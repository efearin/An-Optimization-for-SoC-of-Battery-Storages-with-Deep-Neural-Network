import tensorflow as tf
import os
import pandas as pd
import numpy as np

class Load_Dataset:
    def __init__(self, data_path):

        rawdata = self._load_data(os.path.join(data_path, 'load.csv'))
        self.train_data, self.valid_data, self.test_data = self._split(rawdata)
        self.vocab_size = len(set(self.train_data))

    def _split(self, rawdata):
        rawdata_size = rawdata.shape[0]
        train_size = int(np.floor(rawdata_size * 0.8))
        valid_size = test_size = int(np.floor(rawdata_size * 0.1))

        train_data = rawdata.iloc[:train_size, 'actual'].values
        valid_data = rawdata.iloc[train_size + 1: train_size + 1 + valid_size, 'actual'].values
        test_data = rawdata.iloc[train_size + valid_size + 1:, 'actual'].values
        return train_data, valid_data, test_data

    def _load_data(self, file_fullpath):
        return pd.read_csv(file_fullpath)

class Load_Input:
    def __init__(self, config, data, name=None):
        self.batch_size = config.batch_size
        self.num_steps = config.num_steps
        self.epoch_size = ((len(data) // self.batch_size) - 1) // self.num_steps
        self.input_data, self.targets = self.produce(data, self.batch_size, self.num_steps, name=name)

    def produce(self, data, batch_size, num_steps, name=None):
        with tf.name_scope(name, "LoadProducer", [data, batch_size, num_steps]):
            raw_data = tf.convert_to_tensor(data, name="raw_data", dtype=tf.float32)
            data_len = tf.size(raw_data)
            batch_len = data_len // batch_size
            data = tf.reshape(raw_data[0: batch_size * batch_len],
                              [batch_size, batch_len])
            epoch_size = (batch_len - 1) // num_steps
            assertion = tf.assert_positive(epoch_size,
                                           message="epoch_size == 0, decrease batch_size or num_steps")
            with tf.control_dependencies([assertion]):  # check assertion
                epoch_size = tf.identity(epoch_size,
                                         name="epoch_size")  # return a tensor with the same shape as epoch_size
            # range_input_producer(limit,num_epochs=None,shuffle=True,seed=None,capacity=32,shared_name=None,name=None)
            # Produces the integers from 0 to limit-1 in a queue.
            i = tf.train.range_input_producer(limit=epoch_size, shuffle=False).dequeue()
            # todo: check i

            # strided_slice(input_,begin,end,strides=None,
            #               begin_mask=0,end_mask=0,ellipsis_mask=0,new_axis_mask=0,shrink_axis_mask=0,
            #               var=None,name=None)
            x = tf.strided_slice(input_=data,
                                 begin=[0, i * num_steps],
                                 end=[batch_size, (i + 1) * num_steps])
            x.set_shape([batch_size, num_steps])  # specify tensor shape with much precision

            y = tf.strided_slice(input_=data,
                                 begin=[0, i * num_steps + 1],
                                 end=[batch_size, (i + 1) * num_steps + 1])
            y.set_shape([batch_size, num_steps])  # specify tensor shape with much precision

            return x, y

DATAPATH = "../input"

load_dataset = Load_Dataset(data_path=DATAPATH)
print()