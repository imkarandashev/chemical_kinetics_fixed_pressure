import torch
import torch.nn as nn


def make_dense_leaky_relu(latent_dim, slope):
    return nn.Sequential(nn.Linear(latent_dim, latent_dim), 
                         nn.LeakyReLU(slope))

class UNet_block(nn.Module):
    def __init__(self, latent_dim, slope, middle_block):
        super(UNet_block, self).__init__()
        self.slope = slope
        self.activation = nn.LeakyReLU(slope)
        self.in_block = make_dense_leaky_relu(latent_dim, slope)
        self.out_block = make_dense_leaky_relu(latent_dim, slope)
        self.middle_block = middle_block

    def forward(self, x):        
        identity = x
        x = self.activation(self.in_block(x))
        x = self.activation(self.middle_block(x))
        x = self.activation(self.out_block(x))
        return x + identity

class UNet(nn.Module):
    def __init__(self, t_prev=1, t_next=1, 
                 input_embedding=None,
                 output_embedding=None,
                 latent_dim=None,
                 slope=0.15,
                 level_number=1
                ):
        """
        Шаблон нейронной сети, принимающей на вход t_prev шагов времени 
        и предсказывающая на t_next шагов вперёд.
        Сеть состоит из трёх блоков:
        1) Входной блок (input_embedding)
        2) Промежуточный блок c полносвязной Unet архитектурой
        3) Выходной блок (output_embedding)
        """
        super(UNet, self).__init__()
        
        self.t_prev = t_prev
        self.t_next = t_next
        self.slope = slope
        self.latent_dim = latent_dim
        self.level_number = level_number
        
        self.input_embedding = input_embedding
        self.output_embedding = output_embedding
        
        if level_number == 0:
            #структура сети максимально проста - один слой
            self.unet = make_dense_leaky_relu(self.latent_dim, slope)
        
        elif level_number > 0:
            #создаём unet структуру рекурсивно по иерархии, начиная с самого глубокого
            #самый глубокий уровень
            unet_block = UNet_block(latent_dim=self.latent_dim, slope=0.15, 
                                    middle_block=make_dense_leaky_relu(self.latent_dim, slope))
            #остальные уровни
            for level in range(level_number-1):
                unet_block = UNet_block(latent_dim=self.latent_dim, slope=0.15, 
                                        middle_block=unet_block)
            self.unet = unet_block
        
        else:
            print('error: level_number < 0')
            
    def forward_several_steps(self, x, t_next):
        latent = x
        # запустим сеть в рекуррентном режиме, т.е. несколько раз
        # будем подавать выход на вход
        # для удобства выведем только самый первый и последний шаги
        for t in range(t_next):
            latent = self.forward(latent)
        return latent
    
    def forward_several_steps_and_save(self, x, t_next):
        latent = x
        predictions = []
        # запустим сеть в рекуррентном режиме, т.е. несколько раз
        # будем подавать выход на вход
        # для удобства выведем только самый первый и последний шаги
        for t in range(t_next):
            latent = self.forward(latent)
            predictions.append(latent)
        predictions = torch.stack(predictions, dim=1)
        predictions = predictions.squeeze(dim=2)
        return predictions
        
    def forward(self, x):
        identity = x
        latent = self.input_embedding(x)
        # запустим внутреннюю часть self.unet
        latent = self.unet(latent)
        output = self.output_embedding(latent)
        output += identity[:,:,:-2] 
        #добавим исходные значения и припишем на выход входные значения аргона и азота,
        #которые не должны меняться
        output = torch.cat((output, x[:,:,-2:]), dim=-1)
        #output = F.relu(output)
        return output


