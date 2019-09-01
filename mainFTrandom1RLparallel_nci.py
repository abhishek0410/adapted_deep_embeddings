import argparse
import time
import os
import random
import shutil
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler

#from convertKaggleValset import *
from alexnet import *
#import alexnetOriginal
import dataSplit

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--data', default = '/data/ILSVRC/Data/CLS-LOC/', metavar='DIR', help='path to dataset')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=100, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log_interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')

loadModel = True
resume = False
newSplits = True
B = True
savedFile = 'model_best_fullSplitDS.pth.tar'
valTest = False
setB = True
#splitSize = 50000

def main(noNewLayers, layerActions, epochs, splitSize, gpu, data):
  with torch.cuda.device(gpu):
    global args, best_prec1
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    if(valTest and not(loadModel)):
        model = alexnet(True)
    else:
        model = alexnet()
    if args.cuda:
        model.cuda()
        
    criterion = nn.CrossEntropyLoss()
        
    if(not(valTest) and not(loadModel)):                            
     save_checkpoint({
            'state_dict': model.state_dict(),
            'optimizer' : optimizer.state_dict(),
     }, False, 'randomWeights.pth.tar')


    # Data loading code
    traindir = os.path.join(data, 'ILSVRC/Data/CLS-LOC/train')
    valdir = os.path.join(data, 'ILSVRC/Data/CLS-LOC/val')

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    if(not(valTest)):
     train_dataset = dataSplit.SplitImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]), splitSize = splitSize, val = False, test = False, natvsman = False, setB = True)
     
     """
     if not(newSplits):
        if(splitSize==25000):
            train_idx = np.load('/short/xm00/jp8162/TransferLearning/parallel/Random1/RL/ranBft_25k_train_idx.npy')
            val_idx = np.load('/short/xm00/jp8162/TransferLearning/parallel/Random1/RL/ranBft_25k_val_idx25k.npy')
        elif(splitSize==50000):
            train_idx = np.load('/short/xm00/jp8162/TransferLearning/parallel/Random1/RL/ranBft_50k_train_idx.npy')
            val_idx = np.load('/short/xm00/jp8162/TransferLearning/parallel/Random1/RL/ranBft_50k_val_idx25k.npy')
        elif(splitSize==12500):
            train_idx = np.load('/short/xm00/jp8162/TransferLearning/parallel/Random1/RL/ranBft_12k_train_idx.npy')
            val_idx = np.load('/short/xm00/jp8162/TransferLearning/parallel/Random1/RL/ranBft_12k_val_idx25k.npy')

     else: 
        
        # Define the indices
        indices = list(range(len(train_dataset))) # start with all the indices in training set
        # Random, non-contiguous split
        train_indicies = np.random.choice(indices, size=(.95*len(train_dataset)), replace=False)
        #if(not(splitSize==False)):
        #    train_idx = np.random.choice(train_indicies, size=splitSize, replace=False)
        #else:
        #    train_idx = train_indicies
        val_indicies = [i for i in indices if not(i in train_indicies)] 
        #np.save('./ranB25k_val_indicies', val_indicies)
        val_idx = val_indicies
        #val_indicies = np.load('./ran_val_indicies.npy')
        #valSize = 25000
        #val_idx =  np.random.choice(val_indicies, size=valSize, replace=False)
        #save indicies so we can come back to the same training set
        np.save('./ranBft_50k_train_idx', train_idx)
        np.save('./ranBft_50k_val_idx25k', val_idx)
        #val_idx = np.load('./ranBft_25k_val_idx25k.npy')
     """   

    val_dataset = dataSplit.SplitImageFolder(traindir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]), splitSize=splitSize, val = True, test = False, natvsman = False, setB = True)
    
    """
    if(not(valTest)):
     if resume:
        test_idx = np.load('val_Bft_idx.npy')
     else:
        test_indices = list(range(len(val_dataset)))
        test_split = 5000
        test_idx = np.random.choice(test_indices, size=test_split, replace=False)	
        np.save('val_Bft_idx', test_idx)
    else:
        test_idx = list(range(len(val_dataset)))
    """
    #test_idx = np.load('val_idx.npy')

    if(not(valTest)):
     train_sampler = None
     #train_sampler = SubsetRandomSampler(train_idx)
     train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=8, pin_memory=True)

    #test_sampler = SubsetRandomSampler(val_idx)
    val_loader = torch.utils.data.DataLoader(val_dataset,
        batch_size=args.batch_size, shuffle=False, num_workers=8,
        pin_memory=True)
    """
    if args.evaluate:
        validate(val_loader, model, criterion)
        return
    """

    if(not(valTest)):
      # optionally resume from a checkpoint
      if loadModel:
        if os.path.isfile(savedFile):
            checkpoint = torch.load(savedFile)
            if (resume or B):
                args.start_epoch = checkpoint['epoch']
                best_prec1 = checkpoint['best_prec1']
                model.load_state_dict(checkpoint['state_dict'])
            #optimizer.load_state_dict(checkpoint['optimizer'])

        else:
            exit()
      params_dict = dict(model.named_parameters())
      layers  = ['cl', 'fc2', 'fc1', 'conv5', 'conv4', 'conv3', 'conv2', 'conv1']
      for i in range(0, noNewLayers):
        getattr(model, layers[i]).reset_parameters()
      params = []
      AlexnetKeys = ['cl.weight', 'cl.bias', 'fc2.weight', 'fc2.bias', 'fc1.weight', 'fc1.bias', 'conv5.weight', 'conv5.bias', 'conv4.weight', 'conv4.bias', 'conv3.weight', 'conv3.bias', 'conv2.weight', 'conv2.bias', 'conv1.weight', 'conv1.bias']
      for key, value in reversed(list(params_dict.items())):
        params += [{'params':[value],'lr':layerActions[int(AlexnetKeys.index(key)/2)][0]}]

      optimizer = torch.optim.SGD(params,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
      
      best_prec1 = 0
      no_not_min = 0
      best_val_loss = 7.0                          
      for epoch in range(1, epochs + 1):
        adjust_learning_rate(optimizer, epoch, layerActions)

        # train for one epoch
        train_loss = train(train_loader, model, criterion, optimizer, epoch)

        # evaluate on validation set
        prec1, val_loss = validate(val_loader, model, criterion)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        is_best_val_loss = (val_loss + 0.001) < best_val_loss
        best_val_loss = min(val_loss, best_val_loss)
        if(not(is_best_val_loss)):
           no_not_min+=1
        else:
           no_not_min = 0
        if(no_not_min>3):
            return best_prec1, epoch

        if((epoch>2)&(val_loss>6.0)):
            return best_prec1, epoch        
        if((epoch>5)&(val_loss>5.0)):
            return best_prec1, epoch  
        if((epoch>10)&(val_loss>4.0)):
            return best_prec1, epoch
        if((epoch>15)&(val_loss>3.0)&(splitSize>25000)):
            return best_prec1, epoch
        if((epoch>30)&(val_loss>3.0)&(splitSize>16000)):
            return best_prec1, epoch
        if((epoch>30)&(val_loss>2.8)&(splitSize>25000)):
            return best_prec1, epoch
        if((epoch>30)&(val_loss>2.5)&(splitSize>50000)):
            return best_prec1, epoch
        if((epoch>40)&(val_loss>2.5)&(splitSize>25000)):
            return best_prec1, epoch

    else:
       for epoch in range(1, args.epochs + 1):
          #test_idx = np.random.choice(test_indices, size=test_split, replace=False)	
          #test_sampler = SubsetRandomSampler(test_idx)
          val_loader = torch.utils.data.DataLoader(val_dataset,
               batch_size=args.batch_size, shuffle=False,
               pin_memory=True)
          prec1 = validate(val_loader, model, criterion)
    return best_prec1, epoch

def train(train_loader, model, criterion, optimizer, epoch):
    #f = open('./Random1/RL/50k/ftRanAtoB_fullA_{splitSize}B_lr{FTlr}_new{noNewLayers}_highLR{noHighLR}_{lrDecayEpochs}epochs.txt'.format(FTlr = FTlr, noNewLayers = noNewLayers, noHighLR = noHighLR, lrDecayEpochs = lrDecayEpochs, splitSize = splitSize), 'a')
    #f  = open('childResults.txt', 'a')
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.cuda:
            input = input.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

        # compute output
        output = model(input)
        loss = criterion(output, target)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if i % args.log_interval == 90:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})\n'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, top5=top5))

    return losses.avg                   


def validate(val_loader, model, criterion):
    #f = open('./Random1/RL/50k/ftRanAtoB_fullA_{splitSize}B_lr{FTlr}_new{noNewLayers}_highLR{noHighLR}_{lrDecayEpochs}epochs.txt'.format(FTlr = FTlr, noNewLayers = noNewLayers, noHighLR = noHighLR, lrDecayEpochs = lrDecayEpochs, splitSize = splitSize), 'a')
    #f  = open('childResults.txt', 'a')
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            data_time.update(time.time() - end)

            if args.cuda:
                input = input.cuda(non_blocking=True)
                target = target.cuda(non_blocking=True)

            # compute output
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(prec1[0], input.size(0))
            top5.update(prec5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            
            if i%50==40:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})\n'.format(
                       i, len(val_loader), batch_time=batch_time, data_time=data_time, loss=losses,
                       top1=top1, top5=top5))

    return top1.avg, losses.avg

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch, layerActions):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    finishedNew = True
    for i, param_group in enumerate(optimizer.param_groups):
        if(epoch == layerActions[int(i/2)][1]):
            param_group['lr'] = param_group['lr'] * 0.1
        else:
            if(epoch > layerActions[int(i/2)][1]):
                if(layerActions[int(i/2)][1]>=layerActions[int(i/2)][2]):
                    if(epoch%layerActions[int(i/2)][2]==0):
                        param_group['lr'] = param_group['lr'] * 0.1
                elif(epoch%layerActions[int(i/2)][2]==layerActions[int(i/2)][1]):
                        param_group['lr'] = param_group['lr'] * 0.1

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    main(noNewLayers, layerActions, epochs, splitSize)
