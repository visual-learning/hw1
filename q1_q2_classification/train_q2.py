import torch
import trainer
from utils import ARGS
from simple_cnn import SimpleCNN
from voc_dataset import VOCDataset
import numpy as np
import torchvision
import torch.nn as nn
import random

def get_fc(inp_dim, out_dim, non_linear='relu'):
    """
    Mid-level API. It is useful to customize your own for large code repo.
    :param inp_dim: int, intput dimension
    :param out_dim: int, output dimension
    :param non_linear: str, 'relu', 'softmax'
    :return: list of layers [FC(inp_dim, out_dim), (non linear layer)]
    """
    layers = []
    layers.append(nn.Linear(inp_dim, out_dim))
    if non_linear == 'relu':
        layers.append(nn.ReLU())
    elif non_linear == 'softmax':
        layers.append(nn.Softmax(dim=1))
    elif non_linear == 'none':
        pass
    else:
        raise NotImplementedError
    return layers

class ResNet(nn.Module):
    def __init__(self, num_classes) -> None:
        super().__init__()

        self.resnet = torchvision.models.resnet18(weights='IMAGENET1K_V1')

        self.fc = nn.Sequential(*get_fc(1000, num_classes, 'none'))

    def forward(self, x):
        # TODO return unnormalized log-probabilities here
        x = self.resnet(x)
        out = self.fc(x)
        return out


if __name__ == "__main__":
    np.random.seed(0)
    torch.manual_seed(0)
    random.seed(0)

    # create hyperparameter argument class
    # We will use a size of 224x224 for the rest of the questions. 
    # Note that you might have to change the augmentations

    # TODO experiment a little and choose the correct hyperparameters
    # You should get a map of around 50 in 50 epochs
    args = ARGS(
        epochs=50,
        inp_size=224,
        use_cuda=True,
        val_every=70
        lr=0.01,
        batch_size=64,
        step_size=30,
        gamma=0.1
    )
    
    print(args)

    # TODO define a ResNet-18 model (refer https://arxiv.org/pdf/1512.03385.pdf) 
    # Initialize this model with ImageNet pre-trained weights (except the last layer)
    # You are free to use torchvision.models 
    
    model = ResNet(len(VOCDataset.CLASS_NAMES)).to(args.device)


    # initializes Adam optimizer and simple StepLR scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    # trains model using your training code and reports test map
    test_ap, test_map = trainer.train(args, model, optimizer, scheduler)
    print('test map:', test_map)
