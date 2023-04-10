from matplotlib import pyplot as plt
import torch
import numpy as np

def plot_losses(train_loss, test_loss, NETWORK_NAME, show=True):
    fig, axs = plt.subplots(1, 3, figsize=(15,3))
    for i, loss_type in enumerate(['MSE', 'LH', 'LO']):
        axs[i].plot(train_loss[loss_type], 'r', label='train')
        axs[i].plot(test_loss[loss_type], 'b', label='test')
        axs[i].set_yscale('log')
        axs[i].set_title(loss_type)
        axs[i].legend()
    if show:
        plt.show()
    else:
        fig.savefig(NETWORK_NAME + '_loss_vs_epochs.png', dpi=300)

# Предсказание на N_steps шагов вперёд из начального состояния X0
def predict_on_many_steps(model, X0, N_steps): 
    # Запустим сеть из состояния X0 в цикле, подавая выход сети на вход
#    Xt = X0
#    Prediction_future = []   
#    for i in range(N_steps):
#        #предсказание на много шагов вперёд реккурентно
#        #Xt = model.forward_several_steps(Xt, t_next=1)
#        Xt = model(Xt)
#        Prediction_future.append(Xt)
    Prediction_future = model.forward_several_steps_and_save(X0, t_next=N_steps).squeeze().cpu().detach().numpy()
    #Prediction_future = torch.stack(Prediction_future, dim=1).squeeze().cpu().detach().numpy()
    #Prediction_future = Prediction_future.transpose(1,0,2)
#    Prediction_future = Prediction_future.squeeze().cpu().detach().numpy()
    return Prediction_future

#предсказание на один шаг вперёд
def predict_on_one_step(model, Validation, device, t_prev): 
    # Предсказание на один шаг вперёд из начального состояния X0
    Prediction_one_step = torch.zeros(Validation.shape).to(device)
    Prediction_one_step[:t_prev, :, :] = Validation[:t_prev, :, :]
    for i in range(len(Validation) - t_prev):
        Prediction_one_step[t_prev + i] = model.forward_several_steps(Validation[t_prev + i - 1, :, :].unsqueeze(0), t_next=1)
        #Prediction_one_step[t_prev + i] = model(Validation[t_prev + i - 1, :, :].unsqueeze(0))
    Prediction_one_step = Prediction_one_step.squeeze().cpu().detach().numpy()
    return Prediction_one_step

def one_step_approximation_errors(Validation):
    # Вычислим ошибку, если бы аппроксиматор выдавал ответ в следующем шаге равный значению на входе

    MSE_loss = ((Validation[1:] - Validation[:-1])**2).mean()
    MAE_loss = np.abs(Validation[1:] - Validation[:-1]).mean()

    print("constant_approximation MSE:\t", MSE_loss)
    print("constant_approximation MAE:\t", MAE_loss)
        
    return MSE_loss, MAE_loss

def print_errors(Prediction, Validation):
    # Вычислим ошибки MSE и MAE между Validation и Prediction

    MSE_loss = ((Prediction - Validation)**2).mean()
    MAE_loss = np.abs(Prediction - Validation).mean()

    print("MSE:\t", MSE_loss)
    print("MAE:\t", MAE_loss)
        
    return MSE_loss, MAE_loss

def plot_errors(Prediction, Validation, nsteps, NETWORK_NAME, show=True):

    # Построим графики для сравнения
    fig, axs = plt.subplots(3, 3, figsize=(15,10))
    axs[0,0].plot(Prediction[:nsteps, 0], color='r', label = 'Prediction')
    axs[0,0].plot(Validation[:nsteps, 0], color='b', label = 'Target')
    axs[0,0].set_title('Temperature')
    axs[0,0].legend()
    
    titles = ['T', 'H2', 'O2', 'H2O', 'OH', 'HO2', 'H2O2', 'H', 'O']
    
    for k in range(1, 9):
        i = k // 3
        j = k % 3
        axs[i, j].plot(Prediction[:nsteps, k], color='r', label = 'Prediction')
        axs[i, j].plot(Validation[:nsteps, k], color='b', label = 'Target')
        axs[i, j].set_title(titles[k])
        axs[i, j].legend()
    plt.tight_layout()
    if show:
        plt.show()
    else:
        fig.savefig(NETWORK_NAME + '_Prediction_future_components.png', dpi=300)
        
        
def validate_prediction(model, STEPS_PREV, valid_set, validation_experiment, device, NETWORK_NAME, nsteps, show=True):    
    # Берём один эксперимент из данных из валидационной выборки  
    Validation = torch.FloatTensor(valid_set.data[validation_experiment]).to(device).unsqueeze(1)
    
    # Стартовое состояние X0
    X0 = Validation[:STEPS_PREV, :, :]
    N_steps = len(Validation) - 1
    Prediction_future = predict_on_many_steps(model=model, X0=X0, N_steps=nsteps)
    Prediction_one_step = predict_on_one_step(model=model, Validation=Validation, device=device, t_prev=1)
    Validation = Validation.squeeze().cpu().detach().numpy()

    one_step_approximation_errors(Validation)
    print('предсказание нейронной сети на один шаг вперёд:')
    print_errors(Prediction_one_step, Validation)
    print('предсказание нейронной сети на много шагов вперёд:')
    print_errors(Prediction_future[:N_steps], Validation[1:])

    plot_errors(Prediction_future, Validation[1:], nsteps=nsteps, NETWORK_NAME=NETWORK_NAME, show=show)
    return Prediction_future, Validation