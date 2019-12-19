import sys
import os
import time
import argparse
import numpy as np

import torch
from torch.utils.data import DataLoader
import torch.optim as optim

from torchsummary import summary

import pipeline
import utils
from evaluate import evaluate


def main(args):
    # General parameters ################################################
    CONFIG_FNAME = args.config_fname
    MODE = args.mode
    EXP_NAME = CONFIG_FNAME[:-4]
    print('Make sure the current running path(os.getcwd()) is code directory')
    print('experiment: %s' % EXP_NAME)

    # Devices ################################################
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Using device: %s' % device)

    # Read config files ################################################
    experiment_configs = pipeline.read_config(CONFIG_FNAME)
    if len(experiment_configs) == 1:
        print('Single case in experiment')
    else:
        print('Multiple case in experiment. It might take long time.')
    
    # Folder path for experiment result
    exp_result_dir = '../exp_results/' + EXP_NAME

    # Make folder to save experiment result
    if os.path.exists(exp_result_dir):
        raise(ValueError('This code do not support continue training to prevent overwriting.\n\
            There already exists experiment folder having same name.\n\
            Change the experiment name and start new experiment.'))
    else: os.mkdir(exp_result_dir)

    test_scores = []
    # Run Experiments ################################################
    for num in range(len(experiment_configs)):
        config = experiment_configs[num]
        print('# Case %d ###############################################' % config['case_num'])
        case_dir = exp_result_dir + '/%d' % config['case_num']
        ckpt_path = os.path.join(case_dir, 'ckpt.pt')
        
        # Initialize model
        model = pipeline.select_model(config['model'])
        model = torch.nn.DataParallel(model)
        model.to(device)

        # Optimizer
        optimizer = pipeline.select_optimizer(config['optimizer'], model.parameters(), config['learning_rate'], config['l2_decay'])
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=config['lr_step'], gamma=config['lr_gamma'])

        # Loss function
        loss_fn = pipeline.select_loss(config['loss_fn'])

        ################################################################################################
        # TRAIN & VALID ################################################
        if MODE == 'train':
            # Make foler to save this case
            if os.path.exists(case_dir):
                raise(ValueError('This code do not support continue training to prevent overwriting.\n\
                    There already exists case folder having same name.\n\
                    Change the case number and start new experiment.'))
            else: os.mkdir(case_dir)
            
            # Dataset
            dataset_train = pipeline.select_dataset(config['dataset'],'train')
            dataset_valid = pipeline.select_dataset(config['dataset'],'valid')
            
            # Dataloader
            dataloader_train = DataLoader(dataset_train, batch_size=config['batch_size'], shuffle=True, num_workers=4)
            dataloader_valid = DataLoader(dataset_valid, batch_size=config['batch_size'], shuffle=True, num_workers=4)

            # Initialize variables ------------------------------------------------
            it = 0
            LOG_STEP = 200 # May be adjusted
            train_losses = []
            valid_losses = []
            valid_scores = []
            best_score = -np.inf
            # TODO: Tensorboard
            print('------------------------------------------------')

            for epoch in range(config['max_epoch']):
                one_epoch_start = time.time()
                # Train ------------------------------------------------
                model.train()
                for inputs, labels in dataloader_train:
                    it += 1
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    train_loss = loss_fn(outputs, labels)
                    optimizer.zero_grad()
                    train_loss.backward()
                    optimizer.step()
                    if it % LOG_STEP == 0:
                        print('Case:{}, Epoch:{}, Iter:{}, TrainLoss:{:.4f}'.format(config['case_num'],epoch,it,train_loss))
                        train_losses.append([it, train_loss])
                        # TODO: Tensorboard
                scheduler.step()

                # Valid ------------------------------------------------
                valid_loss, valid_score = evaluate(dataloader_valid, device, model, loss_fn, config['eval_metric'])
                valid_losses.append([it, valid_loss])
                valid_scores.append([it, valid_score])
                # TODO: Tensorboard

                saved = ''
                if valid_score > best_score:
                    best_score = valid_score
                    saved = utils.save_ckpt(ckpt_path, model, optimizer, best_score)
                
                one_epoch_elapsed = time.time()-one_epoch_start
                print('Case:{}, Epoch:{}, ValidScore:{:.4f}, time:{:.2f}s {}'.format(config['case_num'], epoch, valid_score, one_epoch_elapsed, saved))
                print('------------------------------------------------')
            print('Training Finshed!')

        elif MODE == 'test':
            print('No train. Directly go to test mode')

        else:
            raise(ValueError('No such mode'))
        
        # Save Train log
        utils.save_train_log(case_dir, train_losses, valid_losses, valid_scores)
        
        ################################################################################################
        # TEST ################################################
        # Load this case        
        case_dir = exp_result_dir + '/%d' % config['case_num']
        ckpt_path = os.path.join(case_dir, 'ckpt.pt')
        model, optimizer, best_score = utils.load_ckpt(ckpt_path, model, optimizer, best_score)

        # Dataset
        dataset_test = pipeline.select_dataset(config['dataset'],'test')
        
        # Dataloader
        dataloader_test = DataLoader(dataset_test, batch_size=config['batch_size'], shuffle=True, num_workers=4)

        # Test
        test_loss, test_score = evaluate(dataloader_test, device, model, loss_fn, config['eval_metric'])
        print('Case:{}, TestScore:{:.4f}'.format(config['case_num'], test_score))
        test_scores.append([config['case_num'], test_score])

        print('################################################')

    print('################################################')
    print('########      Experiment Finished!      ########')
    print('################################################')

    utils.save_exp_result(exp_result_dir, test_scores)


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_fname', type=str, default='sample.csv')
    parser.add_argument('--mode', type=str, default='train')
    args = parser.parse_args(sys.argv[1:])
    main(args) 


