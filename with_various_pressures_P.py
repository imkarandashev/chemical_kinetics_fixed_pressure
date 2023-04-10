#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import time
import os

import torch.nn.functional as F

from utils.dataset_utils import ChemistryData, transform_data, log_normalize_data, inverse_log_normalize_data, inverse_transform_data

from utils.net_utils import UNet

from utils.learning_utils import compute_weighted_loss_on_several_predictions, process_one_epoch,                                  append_dict_to_dict_of_list

from utils.plot_utils import plot_losses, predict_on_many_steps, predict_on_one_step,                              one_step_approximation_errors, print_errors, plot_errors, validate_prediction

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=512, help='the batch size (default 512)')
parser.add_argument('--latent_dim', type=int, default=100, help='the latent dimension (default 100)')
parser.add_argument('--network_name', type=str, default='UNET', help='the network name (default: UNET)')
parser.add_argument('--epochs', type=int, default=100, help='the epochs number (default 100)')
parser.add_argument('--validation_experiment', type=int, default=4, help='the validation experiment count (default 10)')
parser.add_argument('--level_number', type=int, default=1, help='the unet level number (default 1)')
parser.add_argument('--step_next', type=int, default=1, help='the number of steps next (default 1)')
parser.add_argument('--step_prev', type=int, default=1, help='the number of previous steps (default 1)')
args = parser.parse_args()
print(args)

# In[2]:


# Параметры
BATCH_SIZE = args.batch_size
LATENT_DIM = args.latent_dim
LEVEL_NUM = args.level_number
STEPS_PREV = args.step_prev # количество шагов, по которым будем предсказывать
STEPS_NEXT = args.step_next # количество шагов, на которое будем предсказывать
NETWORK_NAME = f'{args.network_name}_Levels{LEVEL_NUM}_LatentDim{LATENT_DIM}_StepsNext{STEPS_NEXT}'
EPOCHS_NUM = 100 #args.epochs
validation_experiment = args.validation_experiment

device = torch.device("cuda:0" if torch.cuda.is_available () else "cpu")
print(device, torch.cuda.is_available())


# ## Обучение модели

# In[3]:


# считывание данных
train_set = ChemistryData(filename='data/train_set.npy', t_prev=STEPS_PREV, t_next=STEPS_NEXT, transforms=transform_data, coeffs=None)
test_set = ChemistryData(filename='data/test_set.npy', t_prev=STEPS_PREV, t_next=STEPS_NEXT, transforms=transform_data, coeffs=train_set.coeffs)
valid_set = ChemistryData(filename='data/valid_set.npy', t_prev=STEPS_PREV, t_next=STEPS_NEXT, transforms=transform_data, coeffs=train_set.coeffs)

# создание загрузчиков данных для удобства обучения
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=True)
valid_loader = DataLoader(valid_set, batch_size=BATCH_SIZE, shuffle=True)

print(len(train_set), len(test_set), len(valid_set))
print(len(train_loader), len(test_loader), len(valid_loader))

# создание модели нейронной сети
input_dim = 11
output_dim = 9 #последние два копируются из входа
latent_dim = LATENT_DIM
model = UNet(t_prev=STEPS_PREV, t_next=STEPS_NEXT,
             #input_embedding = nn.Sequential(nn.Linear(input_dim, LATENT_DIM), nn.InstanceNorm1d(LATENT_DIM)),
             #input_embedding = nn.Sequential(nn.Linear(input_dim, LATENT_DIM), Flatten_and_Batchnorm(LATENT_DIM)),
             input_embedding = nn.Sequential(nn.Linear(input_dim, LATENT_DIM)), #nn.LeakyReLU(0.15)),
             output_embedding = nn.Sequential(nn.Linear(LATENT_DIM, output_dim)),
             latent_dim=LATENT_DIM,
             slope=0.15,
             level_number=LEVEL_NUM
        ).to(device)
#MSE = nn.MSELoss(reduction='mean')


# ## Обучение

# In[4]:


# оптимизатор
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, verbose=True)

train_loss_history = {'MSE_1': [], 'MSE_t': [], 'MSE': [], 'LH': [], 'LO': [], 'tot': []}
test_loss_history = {'MSE_1': [], 'MSE_t': [], 'MSE': [], 'LH': [], 'LO': [], 'tot': []}

coeffs = {name : torch.FloatTensor(coef).to(device) for name, coef in train_set.coeffs.items()}
for epoch in range(0, EPOCHS_NUM):
    # обучение сети
    mean_loss_record = process_one_epoch(model=model, loader=train_loader, optimizer=optimizer,
                                         device=device, coeffs=coeffs, is_train=True, 
                                         is_LO_LH_compute=False, is_LO_LH_active=False)
    train_loss_history = append_dict_to_dict_of_list(train_loss_history, mean_loss_record)
    
    # проверка на тестовой выборке
    mean_loss_record = process_one_epoch(model=model, loader=test_loader, optimizer=None,
                                         device=device, coeffs=coeffs, is_train=False, 
                                         is_LO_LH_compute=True, is_LO_LH_active=False)
    test_loss_history = append_dict_to_dict_of_list(test_loss_history, mean_loss_record)
    
    lr_scheduler.step(test_loss_history['tot'][-1])
    
    # вывод значений ошибок каждую эпоху
    print(f'epoch {epoch}', 
                 '\n train:', ', '.join([f'{v[0]} {v[1][-1]:0.8f}' for v in train_loss_history.items()]),
                 '\n test: ', ', '.join([f'{v[0]} {v[1][-1]:0.8f}' for v in test_loss_history.items()]))
    
    # сохранение результатов и обученной модели в файл
    if not os.path.isdir('./checkpoints'):
        os.makedirs('./checkpoints')
    if not os.path.isdir('./checkpoints/{}'.format(NETWORK_NAME)):
        os.makedirs('./checkpoints/{}'.format(NETWORK_NAME))
    torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss_history,
                'test_loss': test_loss_history
                }, './checkpoints/{}/{}_epoch_{}_LH_LO.pth'.format(NETWORK_NAME, NETWORK_NAME, epoch))


# In[5]:


plot_losses(train_loss_history, test_loss_history, NETWORK_NAME, show=False)


# In[9]:


prediction, validation = validate_prediction(model, STEPS_PREV, valid_set, validation_experiment, device, NETWORK_NAME, nsteps=500, show=False)


# In[ ]:





# In[ ]:




