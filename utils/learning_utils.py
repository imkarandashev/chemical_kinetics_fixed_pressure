import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from .dataset_utils import inverse_transform_data

MSE = nn.MSELoss(reduction='mean')
MAE = nn.L1Loss(reduction='mean')
# "H2", "O2", "H2O", "OH", "HO2", "H2O2", "H", "O","N2", "Ar"

def LH(predicted, target, device='cuda'):
    """
    Лосс функция, проверяющая сохранение количества атомов водорода в смеси.
    Первый и последние два коэффициента равны нулю, т.к. соответствуют температуре, азоту и аргону
    """
    mult = torch.tensor([0.0, 2.0, 0.0, 2.0, 1.0, 1.0, 2.0, 1.0, 0.0, 0.0, 0.0], device = device, requires_grad = False)
    return torch.mean(torch.abs(torch.matmul((predicted - target), mult)))

def LO(predicted, target, device='cuda'):
    """
    Лосс функция, проверяющая сохранение количества атомов кислорода в смеси.
    Первый и последние два коэффициента равны нулю, т.к. соответствуют температуре, азоту и аргону
    """
    mult = torch.tensor([0.0, 0.0, 2.0, 1.0, 1.0, 2.0, 2.0, 0.0, 1.0, 0.0, 0.0], device = device, requires_grad = False)
    return torch.mean(torch.abs(torch.matmul((predicted - target), mult)))


def compute_weighted_loss_on_several_predictions(predictions, target, device, weights=None, loss=None):
    assert (predictions.shape == target.shape)

    if weights is None:
        weights = (torch.ones(predictions.shape[1]) / predictions.shape[1]).to(device) #constant weights

    if loss is None:
        loss = nn.MSELoss(reduction='mean')

    assert (target.shape[1] == len(weights))
    weighted_loss = 0
    for i in range(len(weights)):
        weighted_loss += weights[i] * loss(predictions[:, i, :], target[:, i, :])

    return weighted_loss


# обучение сети на одну эпоху
def process_one_epoch(model, loader, optimizer, device, coeffs, is_train=False,
                      is_LO_LH_compute=False, is_LO_LH_active=False, dL = 1.0
                      ):
    if is_train:
        model.train()
    else:
        model.eval()

    loss_record = {'MSE_1': [], 'MSE_t': [], 'MSE': [], 'LH': [], 'LO': [], 'tot': []}

    weights = torch.FloatTensor(1 / (1 + np.arange(model.t_next))).to(device) # decreasing weights
    
    MSE = nn.MSELoss(reduction='mean')
    with torch.set_grad_enabled(is_train):
        for input_data, target in tqdm(loader):
            input_data = input_data.to(device)
            target = target.to(device)
            predicted_one_step = model(input_data)
            # predicted_several_step = model.forward_several_steps(input_data, model.t_next)
            several_steps_predictions = model.forward_several_steps_and_save(input_data, model.t_next)

            # среднекваратичная ошибка предсказания
            mse_one_step = MSE(predicted_one_step, target[:, :1, :])
            # mse_several_step = MSE(predicted_several_step, target[:,-1:,:])
            mse_several_step = compute_weighted_loss_on_several_predictions(several_steps_predictions,
                                                                            target,
                                                                            device,
                                                                            weights=weights,  
                                                                            loss=MSE)
            mse = mse_several_step
            # полная функция ошибки
            loss = mse + 0  # + lh + lo

            if is_LO_LH_compute or is_LO_LH_active:
                # проверка сохранения числа атомов водорода и кислорода
                X = inverse_transform_data(several_steps_predictions, coeffs)
                Y = inverse_transform_data(target, coeffs)
                lo = compute_weighted_loss_on_several_predictions(X, Y, device, loss=LO)
                lh = compute_weighted_loss_on_several_predictions(X, Y, device, loss=LH)
                if is_LO_LH_active:
                    loss += dL * (lh + lo)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            loss_record['MSE_1'].append(mse_one_step.item())
            loss_record['MSE_t'].append(mse_several_step.item())
            loss_record['MSE'].append(mse.item())
            loss_record['tot'].append(loss.item())
            if is_LO_LH_compute or is_LO_LH_active:
                loss_record['LH'].append(lh.item())
                loss_record['LO'].append(lo.item())

    mean_loss_record = {key: np.mean(loss_record[key]) for key in loss_record.keys()}
    return mean_loss_record


def combine_two_dicts_of_list(dlist1, dlist2):
    d = {key: dlist1[key] + dlist2[key] for key in dlist1.keys()}
    return d


def append_dict_to_dict_of_list(dlist1, dict2):
    d = {key: dlist1[key] + [dict2[key]] for key in dlist1.keys()}
    return d