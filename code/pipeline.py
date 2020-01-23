import os
import csv

import torch.optim as optim
import torch.nn as nn

from datasets.dataset_predefined import cifar10,mscoco
from datasets.custom1 import *

from models import *

from evaluate import *


################################################
def read_config(config_fname):
    configs = []
    with open('../exp_configs/%s'%config_fname) as config_file:
        reader = csv.DictReader(config_file)
        keys = reader.fieldnames
        for r in reader:
            r[keys[0]] = int(r[keys[0]]) # case_num
            r[keys[3]] = int(r[keys[3]]) # batch_size
            r[keys[4]] = int(r[keys[4]]) # max_epoch
            r[keys[6]] = float(r[keys[6]]) # learning_rate
            r[keys[7]] = int(r[keys[7]]) # lr_step
            r[keys[8]] = float(r[keys[8]]) # lr_gamma
            r[keys[9]] = float(r[keys[9]]) # l2_decay
            r[keys[11]] = bool(r[keys[11]]) # use_tensorboard
            configs.append(r)

    return configs


################################################
def select_model(model_name):
    if model_name == 'vgg13':
        model = vgg(13)
    elif model_name == 'vgg16':
        model = vgg(16)
    elif model_name == 'resnet18':
        model = resnet(18)
    elif model_name == 'resnet34':
        model = resnet(34)
    elif model_name == 'senet18':
        model = senet(18)
    elif model_name == 'densenet':
        model = densenet(121)
    elif model_name == 'mymodel1':
        model = mymodel1()
    else:
        raise(ValueError('No such model'))
    
    return model


################################################
CIFAR10_ROOT_DIR = '../../data/cifar10'
MSCOCO_ROOT_DIR = ''
MSCOCO_ANNOT_PATH = ''
CUSTOM_ROOT_DIR = ''

def select_dataset(dataset_name, dataset_type):
    if dataset_name == 'cifar10':
        dataset = cifar10(CIFAR10_ROOT_DIR,dataset_type,False)
    elif dataset_name == 'mscoco':
        dataset = mscoco(MSCOCO_ROOT_DIR,MSCOCO_ANNOT_PATH,dataset_type)
    elif dataset_name == 'custom1':
        dataset = custom1(CUSTOM_ROOT_DIR,dataset_type)
    else:
        raise(ValueError('No such dataset'))
    
    return dataset


################################################
def select_optimizer(optimizer_name, what_to_optim, learning_rate, l2_decay):
    if optimizer_name == 'adam':
        optimizer = optim.Adam(what_to_optim, lr=learning_rate, weight_decay=l2_decay)
    elif optimizer_name == 'sgd':
        optimizer = optim.SGD(what_to_optim, lr=learning_rate, weight_decay=l2_decay)
    elif optimizer_name == 'custom1':
        optimizer = optim.RMSprop(what_to_optim, lr=learning_rate, weight_decay=l2_decay)
    else:
        raise(ValueError('No such optimizer'))
    
    return optimizer


################################################
def select_loss(loss_name):
    if loss_name == 'mse':
        loss_fn = nn.MSELoss()
    elif loss_name == 'l1':
        loss_fn = nn.L1Loss()
    elif loss_name == 'cross_entropy':
        loss_fn = nn.CrossEntropyLoss()
    else:
        raise(ValueError('No such loss function'))
    
    return loss_fn


################################################
if __name__=='__main__':
    exp_config = read_config('temp.csv')
    print(exp_config[0])