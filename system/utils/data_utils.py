import numpy as np
import os
import torch

def read_data(dataset, data_path, idx, is_train=True):
    if is_train:
        train_data_dir = os.path.join(data_path, 'train/')

        train_file = train_data_dir + str(idx) + '.npz'
        with open(train_file, 'rb') as f:
            with np.load(f, allow_pickle=True) as train_data:

                train_data_dict = {
                    'x': train_data['x'],
                    'y': train_data['y']
                }
        return train_data_dict

    else:
        test_data_dir = os.path.join(data_path, 'test/')

        test_file = test_data_dir + str(idx) + '.npz'
        with open(test_file, 'rb') as f:
            with np.load(f, allow_pickle=True) as test_data:

                test_data_dict = {
                    'x': test_data['x'],
                    'y': test_data['y']
                }
        return test_data_dict


def read_client_data(dataset, data_path, idx, is_train=True):

    if is_train:
        train_data = read_data(dataset, data_path, idx, is_train)
        X_train = torch.Tensor(train_data['x']).type(torch.float32)
        y_train = torch.Tensor(train_data['y']).type(torch.int64)
        train_data = [[x, y] for x, y in zip(X_train, y_train)]
        return train_data
    else:
        test_data = read_data(dataset, data_path, idx, is_train)
        X_test = torch.Tensor(test_data['x']).type(torch.float32)
        y_test = torch.Tensor(test_data['y']).type(torch.int64)
        test_data = [[x, y] for x, y in zip(X_test, y_test)]
        return test_data