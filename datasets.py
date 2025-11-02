import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features

class Dataset_Custom(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h'):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]
        # print(cols)
        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            # print(self.scaler.mean_)
            # exit()
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], axis=1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        # print(f"seq_x shape: {seq_x.shape}, seq_y shape: {seq_y.shape}, seq_x_mark shape: {seq_x_mark.shape}, seq_y_mark shape: {seq_y_mark.shape}")

        return seq_x, seq_y

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
    
class Dataset_SMARD(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='SMARD_converted.csv',
                 target='OT', scale=True, offset=56, window_split=96,
                 split_mode='ratio', ratio_train=0.7, ratio_val=0.2,
                 train_start_date="2015-01-01 00:00:00", train_stop_date="2020-12-31 23:45:00",
                 val_start_date="2021-01-01 00:00:00", val_stop_date="2022-12-31 23:45:00",
                 test_start_date="2023-01-01 00:00:00", test_stop_date="2023-12-31 22:45:00"):
        # size [seq_len, label_len, pred_len]
        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.window_split = window_split

        self.root_path = root_path
        self.data_path = data_path
        self.gap = offset

        self.split_mode = split_mode
        self.ratio_train = ratio_train
        self.ratio_val = ratio_val

        self.train_start_date = train_start_date
        self.train_stop_date = train_stop_date
        self.val_start_date = val_start_date
        self.val_stop_date = val_stop_date
        self.test_start_date = test_start_date
        self.test_stop_date = test_stop_date
        
        assert split_mode in ['ratio', 'fixed']

        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]

        border1s, border2s = None, None 
        
        if self.split_mode == 'fixed':
            idx_train_start = df_raw.index[df_raw['date'].astype(str).str[:19] == self.train_start_date][0]
            idx_train_stop = df_raw.index[df_raw['date'].astype(str).str[:19] == self.train_stop_date][0]
            idx_val_start = df_raw.index[df_raw['date'].astype(str).str[:19] == self.val_start_date][0]
            idx_val_stop = df_raw.index[df_raw['date'].astype(str).str[:19] == self.val_stop_date][0]
            idx_test_start = df_raw.index[df_raw['date'].astype(str).str[:19] == self.test_start_date][0]
            idx_test_end = df_raw.index[df_raw['date'].astype(str).str[:19] == self.test_stop_date][0]
            border1s = [idx_train_start, idx_val_start, idx_test_start]
            border2s = [idx_train_stop, idx_val_stop, idx_test_end]
        else:  # ratio-based split
            num_train = int(len(df_raw) * self.ratio_train) 
            num_test = int(len(df_raw) * self.ratio_val) 
            num_vali = len(df_raw) - num_train - num_test
            border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
            border2s = [num_train, num_train + num_vali, len(df_raw)]
        
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]

    def __getitem__(self, index):
        i = index * self.window_split
        s_begin = i
        s_end = s_begin + self.seq_len
        r_begin = s_end + self.gap
        r_end = r_begin + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]

        return seq_x, seq_y

    def __len__(self):
        return (len(self.data_x) - self.seq_len - self.pred_len - self.gap) // self.window_split + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)