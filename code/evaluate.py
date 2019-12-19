import torch
import torch.nn as nn
import torch.functional as F

def class_acc(outputs, labels):
    class_accuarcy = (outputs.argmax(dim=1) == labels).float().mean().item()
    return class_accuarcy

def evaluate(dataloader, device, model, loss_fn, eval_metric):
    model.eval()
    n = 0
    loss = 0.
    score = 0.
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss += loss_fn(outputs, labels)
            if eval_metric == 'loss':
                score += -loss_fn(outputs, labels) * inputs.size(0)
            elif eval_metric == 'class_acc':
                score += class_acc(outputs, labels) * inputs.size(0)
            else:
                raise(ValueError('No such evaluation metric'))
            n += inputs.size(0)
        loss /= n
        score /= n
    return loss, score
