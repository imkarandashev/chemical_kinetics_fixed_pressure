import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import time
import os
from tqdm import tqdm_notebook
import torch.nn.functional as F

def log_normalize_data(X, eps=1e-10):
    return np.log(1 + X / eps)#np.where(X < eps, X / eps, 1 + np.log(X / eps)) #

def transform_data(X, coeffs, eps=1e-10):
    Y = X + 0
    # температура нормализуется в исходном масштабе
    Y[:, 0] = (X[:, 0] - coeffs['mean'][0]) / coeffs['std'][0]
    # остальные величины нормализуются в логарифмическом масштабе
    Y[:, 1:11] = (log_normalize_data(X[:, 1:11]) - coeffs['log_mean'][1:11]) / coeffs['log_std'][1:11]
    return Y

def inverse_log_normalize_data(X, eps=1e-10):
    # inverse operation for X = np.log(1 + X / eps)
    if type(X) == torch.Tensor:
        return eps * (torch.exp(X) - 1)
    else:
        return eps * (np.exp(X) - 1)

def inverse_transform_data(X, coeffs, eps=1e-10):
    # inverse operation for X = (X - coeffs['mean']) / coeffs['std']
    Y = X + 0
    Y = Y.reshape(-1, X.shape[-1])
    # температура нормализуется в исходном масштабе, остальные величины нормализуются в логарифмическом масштабе
    Y[:, 0] = Y[:, 0] * coeffs['std'][0] + coeffs['mean'][0]
    Y[:, 1:11] = inverse_log_normalize_data(Y[:, 1:11] * coeffs['log_std'][1:11] + coeffs['log_mean'][1:11])
    Y = Y.reshape(X.shape)
    return Y

class ChemistryData(Dataset):
    def __init__(self, filename, t_prev=1, t_next=1, transforms=None, coeffs=None):
        """
        filename - файл с данными в формате .npy
        t_prev - количество шагов, по которым будем предсказывать
        t_next - количество шагов, на которое будем предсказывать
        transforms - предобработка данных (нормировка)
        """

        super(ChemistryData, self).__init__()
        self.filename = filename
        self.t_prev = t_prev
        self.t_next = t_next
        
        if not os.path.exists(self.filename):
            print('error: File' + self.filename + 'does not exist')
            raise FileNotFoundError
        self.data = np.load(self.filename)
        self.step_num = self.data.shape[1]
        
        #оставим только температуру и 10 плотностей компонент
        self.data = self.data[:, :, 1:12]
        
        #посчитаем коэффициенты для нормализации
        a = self.data.reshape(-1, 11)
        self.mean = a.mean(axis=0)
        self.std = a.std(axis=0)
        
        b = log_normalize_data(a)
        self.log_mean = b.mean(axis=0)
        self.log_std = b.std(axis=0)

        self.coeffs = coeffs
        if coeffs is None:
            self.coeffs = {'mean':self.mean, 'std':self.std, 'log_mean':self.log_mean, 'log_std':self.log_std}
        
        #применим трансформацию данных и посчитаем нужные коэффициенты
        if transforms is not None:
            for experiment in range(len(self.data)):
                self.data[experiment] = transforms(self.data[experiment], self.coeffs)
        
    def __len__(self):
        return len(self.data) * (self.step_num - (self.t_prev + self.t_next))
    
    def make_indices(self, index):
        """
        функция считает из глобального индекса (index) два номера:
        experiment_index - номер эксперимента
        internal_index - номер строки внутри эксперимента
        """
        # количество шагов, которые могут быть взяты в качестве начала примера:
        steps_in_experiment = self.step_num - (self.t_prev + self.t_next)
        
        experiment_index = index // steps_in_experiment
        internal_index = index % steps_in_experiment + self.t_prev
        
        return experiment_index, internal_index
    
    def __getitem__(self, ind):
        i, j = self.make_indices(ind)
        # i - номер эксперимента, j - номер шага в эксперименте
        
        x = self.data[i, j - self.t_prev : j, :]
        y = self.data[i, j : j + self.t_next, :]
        x = torch.FloatTensor(x)
        y = torch.FloatTensor(y)
        
        return  x, y
