import torch
import torch.nn as nn
import torch.functional as F

# https://github.com/kuangliu/pytorch-cifar/blob/master/models/vgg.py

vgg_cfg = {
    11: [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    13: [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    16: [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    19: [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class vgg(nn.Module):
    def __init__(self,model_param):
        super(vgg,self).__init__()
        self.features = self.construct_vgg(vgg_cfg[model_param])
        self.out_feature = 512 * 1 * 1
        self.classifier = nn.Linear(self.out_feature, 10)

    def construct_vgg(self,cfg):
        layers = []
        input_channel = 3
        for layer in cfg:
            if layer == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [
                    nn.Conv2d(input_channel, layer, kernel_size=3, padding=1),
                    nn.BatchNorm2d(layer),
                    nn.ReLU(inplace=True)]
                input_channel = layer
        return nn.Sequential(*layers)

    def forward(self,x):
        x = self.features(x)
        x = x.view(-1, self.out_feature)
        out = self.classifier(x)
        return out


if __name__=='__main__':
    from torchsummary import summary
    net = vgg(11)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    net.to(device)
    summary(net,(3,32,32))