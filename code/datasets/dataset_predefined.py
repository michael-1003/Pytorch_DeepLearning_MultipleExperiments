import os

import torchvision
from torchvision import transforms


################################################
def cifar10(root_dir, dataset_type, downloaded):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
    print('CIFAR10 Data at %s' % os.path.abspath(root_dir))
    if dataset_type == 'train':
        dataset = torchvision.datasets.CIFAR10(root=root_dir, train=True, download=not(downloaded), transform=transform_train)
    elif dataset_type == 'valid':
        dataset = torchvision.datasets.CIFAR10(root=root_dir, train=False, download=not(downloaded), transform=transform_test)
    elif dataset_type == 'test':
        dataset = torchvision.datasets.CIFAR10(root=root_dir, train=False, download=not(downloaded), transform=transform_test)
    else:
        raise(ValueError('Invalid dataset type (train,valid,test)'))

    return dataset


################################################
def mscoco(root_dir, annot_path, dataset_type):
    transform = transforms.Compose([
        transforms.ToTensor()])
    if dataset_type == 'train':
        train_dir = '' # TODO
        dataset = torchvision.datasets.CocoDetection(train_dir,annot_path,transform=transform)
    elif dataset_type == 'valid':
        valid_dir = '' # TODO
        dataset = torchvision.datasets.CocoDetection(valid_dir,annot_path,transform=transform)
    elif dataset_type == 'test':
        test_dir = '' # TODO
        dataset = torchvision.datasets.CocoDetection(test_dir,annot_path,transform=transform)
    else:
        raise(ValueError('Invalid dataset type (train,valid,test)'))

    return dataset


if __name__=='__main__':
    dataset = cifar10('../data/cifar10','train',False)
    print(len(dataset))
    print('img size: %s' % list(dataset[0][0].size()))
    print('label: %d' % dataset[0][1])