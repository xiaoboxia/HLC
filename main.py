import os
from turtle import pos
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import MultiStepLR
import torch.backends.cudnn as cudnn
import torchvision.models as tv_models
from torch.cuda.amp import GradScaler, autocast
import torch.optim as optim
import argparse, sys
import numpy as np
import datetime

from metrics import test
from voc import *
from coco import *
from utils import *

import warnings
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser('OURS')
parser.add_argument('--n', type=int, default=0, help="No.")
parser.add_argument('--d', type=str, default='output', help="description")
parser.add_argument('--p', type=int, default=0, help="print")
parser.add_argument('--c', type=int, default=10, help="class")
parser.add_argument('--lr', type=float, default=5e-5)
parser.add_argument('--root', type=str, default='data/COCO/')
parser.add_argument('--result_dir', type=str, help='dir to save result txt files', default='/output/standard/')
parser.add_argument('--noise_rate', type=float, help='overall corruption rate, should be less than 1', default=0.4)
parser.add_argument('--noise_type', type=str, help='[pairflip, symmetric, asymmetric]', default='symmetric')
parser.add_argument('--dataset', type=str, help='voc2007/2012, coco', default='2007')
parser.add_argument('--n_epoch', type=int, default=30)
parser.add_argument('--optimizer', type=str, default='Adam')
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--print_freq', type=int, default=300)
parser.add_argument('--num_workers', type=int, default=4, help='how many subprocesses to use for data loading')
parser.add_argument('--epoch_update_start', type=int, default=5)
parser.add_argument('--model_type', type=str, help='[ce, ours]', default='ce')
parser.add_argument('--fr_type', type=str, help='forget rate type', default='type_1')
parser.add_argument('--split_percentage', type=float, help='train and validation', default=0.9)
parser.add_argument('--gpu', type=str, help='ind of gpu', default='0,1')
parser.add_argument('--weight_decay', type=float, help='l2', default=5e-4)
parser.add_argument('--momentum', type=int, help='momentum', default=0.9)
parser.add_argument('--batch_size', type=int, help='batch_size', default=128)
parser.add_argument('--model_name', type=str, default='OURS')
parser.add_argument('--beta', type=float, help='hyper-parameter', default=0.5)
parser.add_argument('--delta', type=float, help='hyper-parameter', default=0.4)
parser.add_argument('--image_size', type=int, help='image_size', default=224)
args = parser.parse_args()


os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
print(args)
# Seed
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

# Hyper Parameters
batch_size = args.batch_size
learning_rate = args.lr

# load dataset
def load_data(args):
    #np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    
    train_transform = transforms.Compose([
            MultiScaleCrop(args.image_size, scales=(1.0, 0.875, 0.75, 0.66, 0.5), max_distort=2),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    
    val_transform= transforms.Compose([
            Warp(args.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    
    
    if args.dataset=='coco':
        train_dataset = COCO2014(args.root,
                                 phase='train', 
                                 noise_type=args.noise_type, 
                                 noise_rate=args.noise_rate, 
                                 transform=train_transform,
                                 split_per=args.split_percentage,
                                 random_seed=args.seed)

        val_dataset = COCO2014(args.root,
                                 phase='val', 
                                 noise_type=args.noise_type, 
                                 noise_rate=args.noise_rate, 
                                 transform=train_transform,
                                 split_per=args.split_percentage,
                                 random_seed=args.seed)

        test_dataset = COCO2014(args.root, 
                                phase='val',
                                transform=val_transform)
    if args.dataset=='2007':
        train_dataset = Voc2007Classification(args.root,
                                 set_name='train', 
                                 noise_type=args.noise_type, 
                                 noise_rate=args.noise_rate, 
                                 transform=train_transform,
                                 split_per=args.split_percentage,
                                 random_seed=args.seed)

        val_dataset = Voc2007Classification(args.root,
                                 set_name='val', 
                                 noise_type=args.noise_type, 
                                 noise_rate=args.noise_rate, 
                                 transform=val_transform,
                                 split_per=args.split_percentage,
                                 random_seed=args.seed)

        test_dataset = Voc2007Classification(args.root, 
                                set_name='test',
                                transform=val_transform)   
    
    
    if args.dataset=='2012':
        train_dataset = Voc2012Classification(args.root,
                                 set_name='train', 
                                 noise_type=args.noise_type, 
                                 noise_rate=args.noise_rate, 
                                 transform=train_transform,
                                 split_per=args.split_percentage,
                                 random_seed=args.seed)

        val_dataset = Voc2012Classification(args.root,
                                 set_name='val', 
                                 noise_type=args.noise_type, 
                                 noise_rate=args.noise_rate, 
                                 transform=val_transform,
                                 split_per=args.split_percentage,
                                 random_seed=args.seed)

        test_dataset = Voc2007Classification(root='data/2007/', 
                                set_name='test',
                                transform=val_transform)  
    

    return train_dataset, val_dataset, test_dataset




def adjust_learning_rate(optimizer, epoch):
    for param_group in optimizer.param_groups:
        param_group['lr'] = alpha_plan[epoch]
        param_group['betas'] = (beta1_plan[epoch], 0.999)

def accuracy(logit, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    output = F.softmax(logit, dim=1)
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


# Train the Model


def train_one_step(net, data, label, optimizer, criterion):
    net.train()
    pred, _ = net(data)
    loss = criterion(torch.sigmoid(pred), label)
    loss.backward()
    nn.utils.clip_grad_norm_(net.parameters(), max_norm=10.0)
    optimizer.step()
    optimizer.zero_grad()

    return loss

def train_one_step_correction(net, data, labels, optimizer, criterion, delta, beta):
    net.train()
    pred, label_dependency = net(data)
    pred = torch.sigmoid(pred)
    corrected_labels_batch = torch.zeros((labels.size(0), labels.size(1)))

    for j in range (pred.size(0)):
        t_pred = pred[j]
        t_number_labels = torch.nonzero(labels[j]).size(0)
        t_noisy_labels = torch.nonzero(labels[j])
        t_predicted_labels = torch.topk(t_pred, int(t_number_labels)).indices
        original_sc = beta * torch.sum(t_pred[t_noisy_labels]) + (1-beta) * label_dependency_capture(label_dependency[j], t_noisy_labels)
        predicted_sc = beta * torch.sum(t_pred[t_predicted_labels]) + ( 1-beta) * label_dependency_capture(label_dependency[j], t_predicted_labels)
        SR = original_sc / predicted_sc
        
        if SR <= delta:
            corrected_labels_batch[j, t_predicted_labels] = 1.
        else:
            corrected_labels_batch[j, t_noisy_labels] = 1.
    
    loss = criterion(pred.cuda(), corrected_labels_batch.cuda())
    loss.backward()
    nn.utils.clip_grad_norm_(net.parameters(), max_norm=10.0)
    optimizer.step()
    optimizer.zero_grad()
    
    return loss, corrected_labels_batch

def train(model, train_loader, optimizer1):
    model.train()
    train_total = 0
    train_loss = 0.
    orginal = []
    for i, (data, labels) in enumerate(train_loader):
        data = data.cuda()
        labels = labels.cuda()
        orginal.append(labels)
        # Forward + Backward + Optimize
        train_total += 1
        loss = train_one_step(model, data.float(), labels.float(), optimizer1, nn.BCELoss())
        train_loss += loss
    train_loss_ = train_loss  / float(train_total)

    return train_loss_, orginal

def train_after_correction(model, train_loader, new_labels, optimizer1, delta, beta):
    model.train()
    train_total = 0
    train_loss = 0.
    corrected_labels = []
    for i, (data, labels) in enumerate(train_loader):
        data = data.cuda()
        labels = new_labels[i].cuda()
        
        # Forward + Backward + Optimize
        train_total += 1
        loss, corrected_labels_batch = train_one_step_correction(model, data.float(), labels.float(), optimizer1, nn.BCELoss(), delta, beta)
        train_loss += loss
        corrected_labels.append(corrected_labels_batch)
    train_loss_ = train_loss  / float(train_total)

    return train_loss_, corrected_labels

def label_dependency_capture(label_dependency, labels):
    posterior_pro_y_y = 0.
    for j in range(labels.size(0)):
        for k in range(labels.size(0)):
            if int(labels[k]) != int(labels[j]):
                t = label_dependency[int(labels[j]), int(labels[k])]
                posterior_pro_y_y += t
    return posterior_pro_y_y





# Evaluate the Model
def evaluate(test_loader, model1):
    
    model1.eval()  # Change model to 'eval' mode.
    correct1 = 0
    total1 = 0
    with torch.no_grad():
        for data, labels, _ in test_loader:
            data = data.cuda()
            logits1 = model1(data)
            outputs1 = F.softmax(logits1, dim=1)
            _, pred1 = torch.max(outputs1.data, 1)
            total1 += labels.size(0)
            correct1 += (pred1.cpu() == labels.long()).sum()

        acc1 = 100 * float(correct1) / float(total1)

    return acc1


def main(args):
    # Data Loader (Input Pipeline)
    print(args)
    # Data Loader (Input Pipeline)
    print('loading dataset...')
    train_dataset, val_dataset, test_dataset = load_data(args)


    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               num_workers=args.num_workers,
                                               drop_last=False,
                                               shuffle=False)
    
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                              batch_size=batch_size,
                                              num_workers=args.num_workers,
                                              drop_last=False,
                                              shuffle=False)
    
    
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size,
                                              num_workers=args.num_workers,
                                              drop_last=False,
                                              shuffle=False)
    
   
    # Define models
    print('building model...')
    import gcn_ours
    
    clf1 = gcn_ours.get_model(num_classes=args.c)
    clf1 = nn.DataParallel(clf1)
    clf1.cuda()
    optimizer1 = torch.optim.Adam(clf1.parameters(), lr=learning_rate)
    epoch = 0
    
    for epoch in range(0, args.n_epoch):
        # train models
        clf1.train()

        delta = args.delta * max(0, 0.2*(10-epoch))
        print(delta)
        print('Training...')
        if epoch < args.epoch_update_start:
            train_loss, corrected_labels = train(clf1, train_loader, optimizer1)  
        else:
            train_loss, corrected_labels = train_after_correction(clf1, train_loader, corrected_labels, optimizer1, delta, args.beta)
        
        val_1, val_2 = test(clf1, val_loader, return_map=True)
        val_loss, v_hloss, v_rloss, cover, avgpre, oneerror, acc = val_1
        map, OP, OR, OF1, CP, CR, CF1 = val_2
        
        test_1, test_2 = test(clf1, test_loader, return_map=True)
        
        val_loss_, v_hloss_, v_rloss_, cover_, avgpre_, oneerror_, acc_ = test_1
        map_, OP_, OR_, OF1_, CP_, CR_, CF1_ = test_2
        
        print('Epoch', epoch, 'val_loss, hloss, rloss, cover, oneerror, avgpre, acc', round(val_loss, 5), round(v_hloss, 5), round(v_rloss, 5), round(cover, 5), round(oneerror, 5), round(avgpre, 5), round(acc, 5))
        print('Epoch', epoch, 'val_map, OP, OR, OF1, CP, CR, CF1', round(map, 5), round(OP, 5), round(OR, 5), round(OF1, 5), round(CP, 5), round(CR, 5), round(CF1, 5))
        print('Epoch', epoch, 'test_loss, hloss, rloss, cover, oneerror, avgpre, acc', round(val_loss_, 5), round(v_hloss_, 5), round(v_rloss_, 5), round(cover_, 5), round(oneerror_, 5),round(avgpre_, 5),round(acc_, 5))
        print('Epoch', epoch, 'test_map, OP, OR, OF1, CP, CR, CF1', round(map_, 5), round(OP_, 5), round(OR_, 5), round(OF1_, 5), round(CP_, 5),round(CR_, 5),round(CF1_, 5))  
        
    
    

if __name__ == '__main__':
    main(args)