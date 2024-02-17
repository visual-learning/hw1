from __future__ import print_function

import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import utils
from voc_dataset import VOCDataset


def save_this_epoch(args, epoch):
    if args.save_freq > 0 and (epoch+1) % args.save_freq == 0:
        return True
    if args.save_at_end and (epoch+1) == args.epochs:
        return True
    return False


def save_model(epoch, model_name, model):
    filename = 'checkpoint-{}-epoch{}.pth'.format(
        model_name, epoch+1)
    print("saving model at ", filename)
    torch.save(model, filename)


def my_sigmoid(x):
    return 1/(1+torch.exp(-x))

EPS = 1e-10

def loss_fn(output, target, wgt):
    '''Multilabel classification loss function'''
    # print("Begin loss_fn")
    # print("output",output.shape)
    # print("target",target.shape)
    # print("wgt",wgt.shape)
    N, num_classes = output.shape
    output_prob = my_sigmoid(output)
    # loss = 0.0
    # for i in range(10):
    #     for j in range(num_classes):
    #         print("output, target, wgt at i, j = ",i,j,output[i][j],target[i][j],wgt[i][j])
    #         loss += (-1/(N*num_classes)) *wgt[i][j] * (target[i][j] * torch.log(output[i][j]) + (1 - target[i][j]) * torch.log(1 - output[i][j]))
    #         print("loss at i, j = ",i,j,loss.item(),(-1/(N*num_classes)) *wgt[i][j] * (target[i][j] * torch.log(output[i][j]) + (1 - target[i][j]) * torch.log(1 - output[i][j])))
        # loss /= num_classes
    # loss /=  -N
    # loss1 = (-1/(N*num_classes))*torch.sum(wgt*(target*torch.log(output) + (1-target)*torch.log(1-output)))
    loss = (-1/(N*num_classes))*torch.sum(wgt*(target*torch.log(output_prob) + (1-target)*torch.log(1-output_prob) + EPS))
    # print("End loss_fn. loss = ",loss.item())
    return loss



def train(args, model, optimizer, scheduler=None, model_name='model'):
    writer = SummaryWriter()
    train_loader = utils.get_data_loader(
        'voc', train=True, batch_size=args.batch_size, split='trainval', inp_size=args.inp_size)
    test_loader = utils.get_data_loader(
        'voc', train=False, batch_size=args.test_batch_size, split='test', inp_size=args.inp_size)

    # Ensure model is in correct mode and on right device
    model.train()
    model = model.to(args.device)

    cnt = 0

    for epoch in range(args.epochs):
        for batch_idx, (data, target, wgt) in enumerate(train_loader):
            data, target, wgt = data.to(args.device), target.to(args.device), wgt.to(args.device)

            optimizer.zero_grad()
            output = model(data)

            ##################################################################
            # TODO: Implement a suitable loss function for multi-label
            # classification. You are NOT allowed to use any pytorch built-in
            # functions. Remember to take care of underflows / overflows.
            # Function Inputs:
            #   - `output`: Outputs from the network
            #   - `target`: Ground truth labels, refer to voc_dataset.py
            #   - `wgt`: Weights (difficult or not), refer to voc_dataset.py
            # Function Outputs:
            #   - `output`: Computed loss, a single floating point number
            ##################################################################
            loss = loss_fn(output, target, wgt)
            ##################################################################
            #                          END OF YOUR CODE                      #
            ##################################################################
            
            loss.backward()
            
            if cnt % args.log_every == 0:
                writer.add_scalar("Loss/train", loss.item(), cnt)
                print('Train Epoch: {} [{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, cnt, 100. * batch_idx / len(train_loader), loss.item()))
                
                # Log gradients
                for tag, value in model.named_parameters():
                    if value.grad is not None:
                        writer.add_histogram(tag + "/grad", value.grad.cpu().numpy(), cnt)

            optimizer.step()
            
            # Validation iteration
            if cnt % args.val_every == 0:
                model.eval()
                ap, map = utils.eval_dataset_map(model, args.device, test_loader)
                print("map: ", map)
                writer.add_scalar("map", map, cnt)
                model.train()
            
            cnt += 1

        if scheduler is not None:
            scheduler.step()
            writer.add_scalar("learning_rate", scheduler.get_last_lr()[0], cnt)

        # save model
        if save_this_epoch(args, epoch):
            save_model(epoch, model_name, model)

    # Validation iteration
    test_loader = utils.get_data_loader('voc', train=False, batch_size=args.test_batch_size, split='test', inp_size=args.inp_size)
    ap, map = utils.eval_dataset_map(model, args.device, test_loader)
    return ap, map
