import os

import numpy as np

import torch


def save_ckpt(ckpt_path, model, optimizer, best_score):
    ckpt = {'model':model.state_dict(),
            'optimizer':optimizer.state_dict(),
            'best_score':best_score}
    torch.save(ckpt,ckpt_path)
    return 'best score'

def load_ckpt(ckpt_path, model, optimizer, best_score):
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path)
        try:
            model.load_state_dict(ckpt['model'])
            optimizer.load_state_dict(ckpt['optimizer'])
            best_score = ckpt['best_score']  
            print('Checkpoint is loaded (best_score:%.4f)' % best_score)
        except RuntimeError as e:
            print('Wrong checkpoint')
    else:
        raise(ValueError('No checkpoint exists'))
    return model, optimizer, best_score


def save_train_log(case_dir, train_losses, valid_losses, valid_scores):
    train_losses = np.array(train_losses)
    valid_losses = np.array(valid_losses)
    valid_scores = np.array(valid_scores)
    
    np.savetxt('%s/train_losses.txt'%case_dir,train_losses,delimiter=',')
    np.savetxt('%s/valid_losses.txt'%case_dir,valid_losses,delimiter=',')
    np.savetxt('%s/valid_scores.txt'%case_dir,valid_scores,delimiter=',')



def save_exp_result(exp_result_dir, test_scores):
    test_scores = np.array(test_scores)
    np.savetxt('%s/test_scores.txt'%exp_result_dir,test_scores,delimiter=',')
