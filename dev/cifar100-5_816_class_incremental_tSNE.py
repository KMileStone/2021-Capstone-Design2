import time
print("start time : {}\n".format(time.time()))



import os
import torch
import torch.nn as nn
import numpy as np
import argparse
import copy
# import data_loader
import pdb
import torch.nn.functional as F
import re, random, collections
import pickle

import math

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns
tsne = TSNE()





import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]='0'





class args():
    data_path = "./Datasets/CIFAR100/"
    num_class = 100
    class_per_task = 10
    num_task = 10
    dataset = "cifar100"
    train_batch = 128
    test_batch = 128
    workers = 16

    random_classes = False
    validation = 0
    overflow = False

    batch_size = 128
    lr = 0.01
    resume = False
    
    total_epoch = 250
    model_path = 'ckpt816/cifar100-10-tSNE-ce' ### change

    #+++ additive angular margin loss +++#
    scale = 32      # exp range[0, 8.8861e+6, 7.8964e+13]
    margin = 0.9396 # cos e, e is 20 degree
    l = 20          # ratio of intra and inter task loss
    #++++++++++++++++++++++++++++++++++++#


args = args()





def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


# seed=torch.randint(1,10000,[1])
set_seed(3473)





import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import argparse
# from utils import progress_bar
from torch.optim.lr_scheduler import MultiStepLR
torch.set_printoptions(precision=5,sci_mode=False)





class ConvAdapt(nn.Module):
    def __init__(self, in_channels, out_channels, p):
        super(ConvAdapt, self).__init__()
        # Groupwise Convolution
        self.gwc = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, groups=int(p/gp), bias=True)
        # Pointwise Convolution
        self.pwc = nn.Conv2d(in_channels, out_channels, kernel_size=1, groups=int(p/pt),bias=True)

    def forward(self, x):
        return self.gwc(x) + self.pwc(x)

'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.lhc1=ConvAdapt(planes,planes,planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.lhc2=ConvAdapt(planes,planes,planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                ConvAdapt(self.expansion*planes,self.expansion*planes,int(self.expansion*planes)),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.lhc1(self.conv1(x))))
        out = self.bn2(self.lhc2(self.conv2(out)))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=args.class_per_task):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.hc1=ConvAdapt(64,64,64)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out=self.conv1(x)
        out = F.relu(self.bn1(out))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        feat = out.view(out.size(0), -1)
        out = self.linear(feat)
        return out,feat


def Net():
    return ResNet(BasicBlock, [2,2,2,2])





def save_model(task, acc, model): #, feat_mean):
    print('Saving..')
    statem = {
        'net': model.state_dict(),
        'acc': acc,

        #+++ check task with mean +++#
        #'feat_mean' : feat_mean,
        #++++++++++++++++++++++++++++#

    }
    fname = args.model_path
    if not os.path.isdir(fname):
        os.makedirs(fname)
    torch.save(statem, fname + '/ckpt_task' + str(task) + '.pth')


def load_model(task, model):
    fname = args.model_path
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    print(fname + '/ckpt_task' + str(task) + '.pth')
    checkpoint = torch.load(fname + '/ckpt_task' + str(task) + '.pth')
    model.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']

    #+++ check task with mean +++#
    #feat_mean = checkpoint['feat_mean']
    #++++++++++++++++++++++++++++#

    return best_acc #, feat_mean





#+++ additive angular margin loss +++#
def get_intra_task_loss(feat_current, feat_current_mean):
    # normalize
    nfeat_current = F.normalize(feat_current)
    nfeat_current_mean = F.normalize(feat_current_mean.view(1, -1).cuda())

    # cosine
    cosine = nfeat_current @ nfeat_current_mean.transpose(0,1)

    # exp
    out = torch.exp(args.scale*(1-cosine)) - 1

    return out.mean()

def get_inter_task_loss(feat_current, feat_prev_mean):
    # normalize
    nfeat_current = F.normalize(feat_current)
    nfeat_prev_mean = F.normalize(feat_prev_mean.view(1, -1).cuda())

    # cosine
    cosine = nfeat_current @ nfeat_prev_mean.transpose(0,1)

    # exp & margin penalty
    out = torch.exp(args.scale*(cosine-args.margin)) - 1

    return out.mean()
#++++++++++++++++++++++++++++++++++++#

### change

### tSNE: CE
def train(train_loader, epoch, task, model, task_model):
    print('\nEpoch: %d' % epoch)
    model.train()
    global best_acc
    train_loss = 0
    correct = 0
    total = 0

    Sloss = 0
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.cuda(), targets.cuda()
        targets = targets - task * args.class_per_task  # assume sequential split for random  split mapping should me changed
        optimizer.zero_grad()
        outputs, feat_current = model(inputs)

        loss = criterion(outputs, targets)

        loss.backward()

        optimizer.step()

        train_loss += loss.item()

        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        acc = 100. * correct / total

    print("[Train: ], [%d/%d: ], [Accuracy: %f], [Loss: %f] [LossS: %f] [Lr: %f]"
          % (epoch, args.total_epoch, acc, train_loss / batch_idx, Sloss / batch_idx,
             optimizer.param_groups[0]['lr']))

### tSNE: CE+FDM
# def train(train_loader, epoch, task, model, task_model):
#     print('\nEpoch: %d' % epoch)
#     model.train()
#     global best_acc
#     train_loss = 0
#     correct = 0
#     total = 0

#     Sloss = 0
#     for batch_idx, (inputs, targets) in enumerate(train_loader):
#         inputs, targets = inputs.cuda(), targets.cuda()
#         targets = targets - task * args.class_per_task  # assume sequential split for random  split mapping should me changed
#         optimizer.zero_grad()
#         outputs, feat_current = model(inputs)

#         loss = criterion(outputs, targets)

#         ################################################################################
#         ############# this is slow version here we dont save \mu and \sigma of KLD######
#         # saving mu and sigma and maximize the distance from the current batch significantly
#         # speedup the model####
#         if task >= 1:
#             for m in range(task):
#                 with torch.no_grad():
#                     task_model[m].eval()
#                     out, feat_prev = task_model[m](inputs)
#                 mean_loss = torch.max(1 - torch.dist(feat_current, feat_prev, 2) / args.train_batch,
#                                       torch.zeros(1).cuda()) / task
#                 loss += 0.05 * mean_loss[0]
#         ###################################################################################

#         loss.backward()

#         optimizer.step()

#         train_loss += loss.item()

#         _, predicted = outputs.max(1)
#         total += targets.size(0)
#         correct += predicted.eq(targets).sum().item()
#         acc = 100. * correct / total

#     print("[Train: ], [%d/%d: ], [Accuracy: %f], [Loss: %f] [LossS: %f] [Lr: %f]"
#           % (epoch, args.total_epoch, acc, train_loss / batch_idx, Sloss / batch_idx,
#              optimizer.param_groups[0]['lr']))

### tSNE: CE+NPC
# def train(train_loader, epoch, task, model, task_model):
#     print('\nEpoch: %d' % epoch)
#     model.train()
#     global best_acc
#     train_loss = 0
#     correct = 0
#     total = 0

#     #+++ additive angular margin loss +++#
#     train_ce_loss = 0
#     train_angular_loss = 0
#     #++++++++++++++++++++++++++++++++++++#

#     #+++ check task with memory +++#
#     #feat_mean_list = []
#     #global feat_mean
#     #++++++++++++++++++++++++++++++#

#     Sloss = 0
#     for batch_idx, (inputs, targets) in enumerate(train_loader):
#         inputs, targets = inputs.cuda(), targets.cuda()
#         targets = targets - task * args.class_per_task  # assume sequential split for random  split mapping should me changed
#         optimizer.zero_grad()
#         outputs, feat_current = model(inputs)

#         #+++ check task with memory +++#
#         #feat_mean_list.append(feat_current.mean(dim=0))         # feat mean per batch
#         #++++++++++++++++++++++++++++++#

#         loss = criterion(outputs, targets)

#         #+++ additive angular margin loss +++#
#         ce_loss = loss
#         angular_loss = get_intra_task_loss(feat_current, feat_current.mean(dim=0))    # left term of NPC

#         if task >= 1:
#             for m in range(task):
#                 with torch.no_grad():
#                     task_model[m].eval()
#                     out, feat_prev = task_model[m](inputs)

#                 #loss += args.l * get_angular_loss(feat_current, feat_prev) / task
#                 angular_loss += args.l * max(get_inter_task_loss(feat_current, feat_prev.mean(dim=0)), 0)   # right term of NPC

#         angular_loss /= task + 1

#         train_ce_loss += ce_loss.item()
#         train_angular_loss += angular_loss

#         loss += angular_loss
#         #++++++++++++++++++++++++++++++++++++#

#         loss.backward()

#         optimizer.step()

#         train_loss += loss.item()

#         _, predicted = outputs.max(1)
#         total += targets.size(0)
#         correct += predicted.eq(targets).sum().item()
#         acc = 100. * correct / total

#     #+++ check task with memory +++#
#     #feat_mean = torch.stack(feat_mean_list).mean(dim=0)   # feat mean per epoch
#     #++++++++++++++++++++++++++++++#

#     print("[Train: ], [%d/%d: ], [Accuracy: %f], [Loss: %f] [ce_loss: %f] [angular_loss: %f] [Lr: %f]"
#           % (epoch, args.total_epoch, acc, train_loss / batch_idx, train_ce_loss / batch_idx, train_angular_loss / batch_idx,
#              optimizer.param_groups[0]['lr']))





def check_task(task, inputs, model, task_model):
    joint_entropy = []
    with torch.no_grad():
        for m in range(task):
            task_model[m].eval()
            outputs, _ = task_model[m](inputs)

            sout = F.softmax(outputs, 1)
            entropy = -torch.sum(sout * torch.log(sout + 0.0001), 1)
            joint_entropy.append(entropy)

        model.eval()
        outputs, _ = model(inputs)

        sout = F.softmax(outputs, 1)
        entropy = -torch.sum(sout * torch.log(sout + 0.0001), 1)
        joint_entropy.append(entropy)

    all_entropy = torch.zeros([joint_entropy[0].shape[0], task + 1]).cuda()
    for i in range(task + 1):
        all_entropy[:, i] = joint_entropy[i]
    ctask = torch.argmin(all_entropy, axis=1) == task
    correct = sum(ctask)

    return ctask, correct, all_entropy





def test(test_loader, task, model, task_model): #, task_feat_mean):
    global best_acc
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    cl_loss = 0
    tcorrect = 0

    #+++ check task with memory +++#
    #global feat_mean
    #++++++++++++++++++++++++++++++#

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets1 = inputs.cuda(), targets.cuda()
            targets = targets1 - task * args.class_per_task
            if task > 0:
                #+++ check task with mean +++#
                #correct_sample, Ncorrect, _ = check_task(task, inputs, model, task_model, task_feat_mean)
                #++++++++++++++++++++++++++++#
                correct_sample, Ncorrect, _ = check_task(task, inputs, model, task_model)
                
                tcorrect += Ncorrect
                inputs = inputs[correct_sample]
                targets = targets[correct_sample]
            if inputs.shape[0] != 0:
                outputs, _ = model(inputs)
                loss = criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                correct += predicted.eq(targets).sum().item()
            total += targets1.size(0)
    if task > 0:
        taskC = tcorrect.item() / total
    else:
        taskC = 1.0
    acc = 100. * correct / total
    print("[Test Accuracy: %f], [Loss: %f] [Correct: %f]" % (acc, test_loss / batch_idx, taskC))

    if acc >= best_acc:
        #+++ check task with mean +++#
        #save_model(task, acc, model, feat_mean)
        #++++++++++++++++++++++++++++#
        save_model(task, acc, model)

        best_acc = acc
    return acc

#+++ known test +++#
def test_known(test_loader, task, model, epoch):
    model.eval()
    test_known_loss = 0
    correct = 0
    total = 0
    epoch_to_plot = [0, 50, 100, 150, 200, 249]

    ### tSNE
    feat_list = []
    targets_list = []
    if epoch in epoch_to_plot:
        fig = plt.figure(figsize = (10,10))
        plt.axis('off')
        sns.set_style('darkgrid')

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets1 = inputs.cuda(), targets.cuda()
            targets = targets1 - task * args.class_per_task

            outputs, feat = model(inputs)

            ### tSNE
            if epoch in epoch_to_plot:
                feat_list.append(feat.cpu())
                targets_list.append(targets.cpu())

            loss = criterion(outputs, targets)

            test_known_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()
            total += targets1.size(0)
    
    ### tSNE
    if epoch in epoch_to_plot:
        feat_all = torch.cat(feat_list, dim=0)
        targets_all = torch.cat(targets_list, dim=0)
        feat_tsne = tsne.fit_transform(feat_all.data)
        sns.scatterplot(feat_tsne[:,0], feat_tsne[:,1], hue=targets_all, legend='full', palette=sns.color_palette("bright", 10))
        plt.legend(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])
        plt.savefig('ce.{}.{}.png'.format(task, epoch))    ### change

    acc = 100. * correct / total
    print("[Test Accuracy known: %f], [Loss: %f] [Correct: %f]" % (acc, test_known_loss / batch_idx, 1.0))
#++++++++++++++++++#





# this false the gradient of the global parameter after the 1st task

def grad_false(model):
    gradf=[0,7,14,21,28,35,42,49,56,63,70,77,84,91,98,105,112,119,126,133]
    i=0
    for p in model.parameters():
        if i in gradf:
            p.requires_grad=False
        i=i+1





import incremental_dataloader as data
inc_dataset = data.IncrementalDataset(
                                dataset_name=args.dataset,
                                args = args,
                                random_order=args.random_classes,
                                shuffle=True,
                                seed=1,
                                batch_size=args.train_batch,
                                workers=args.workers,
                                validation_split=args.validation,
                                increment=args.class_per_task,
                            )
task_data=[]
for i in range(args.num_task):
    task_info, train_loader, val_loader, test_loader = inc_dataset.new_task()

    task_data.append([train_loader,test_loader])





# groupwise and pointwise group convolutional size
gp=8
pt=16
###############################################

#+++ check task with memory +++#
#feat_mean = 0
#task_feat_mean = []
#++++++++++++++++++++++++++++++#

task_model=[]
task_acc=[]
criterion = nn.CrossEntropyLoss()

for task in range(args.num_task):
    #+++ resume process +++#
    if args.resume and os.path.isfile(args.model_path+'/ckpt_task'+str(task)+'.pth'):
        modelm = Net().cuda()
        acc1, feat_mean = load_model(task,modelm)
        task_model.append(copy.deepcopy(modelm))
        task_feat_mean.append(feat_mean)
        continue
    #++++++++++++++++++++++#

    ### dev: skip task 0
    if task==0:
        modelm = Net().cuda()
        acc1 = load_model(task,modelm)
        task_model.append(copy.deepcopy(modelm))
        continue

    if task==0:
        best_acc=0
        print('Training Task :---'+str(task))

        train_loader, test_loader = task_data[task][0],task_data[task][1]
        modelm = Net().cuda()

#         criterion = nn.CrossEntropyLoss()
        mse_loss=nn.MSELoss()
        optimizer = optim.SGD(modelm.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-3)
        schedulerG = MultiStepLR(optimizer, milestones=[100, 150, 200], gamma=0.1)

        #+++ check task with memory +++#
        for epoch in range(args.total_epoch):
            #feat_mean = 0
            train(train_loader,epoch,task,modelm,task_model)
            #test(test_loader,task,modelm,task_model,task_feat_mean)
            test(test_loader,task,modelm,task_model)
            test_known(test_loader,task,modelm,epoch)
            schedulerG.step()

        #acc1, feat_mean = load_model(task,modelm)
        acc1 = load_model(task,modelm)
        task_model.append(copy.deepcopy(modelm))
        #task_feat_mean.append(feat_mean)
        #++++++++++++++++++++++++++++++#


    if task!=0:
        print('Training Task :---'+str(task))
        best_acc=0
        train_loader, test_loader = task_data[task][0],task_data[task][1]
        modelm = Net().cuda()

        #+++ check task with memory +++#
        #acc1, _ = load_model(task-1,modelm)
        acc1 = load_model(task-1,modelm)
        grad_false(modelm)
        #+++++++++++++++++++++++++++++++#

#         criterion = nn.CrossEntropyLoss()
        mse_loss=nn.MSELoss()
        optimizer = optim.SGD(modelm.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-3)
        schedulerG = MultiStepLR(optimizer, milestones=[100,150,200],gamma=0.1)

        #+++ check task with memory +++#
        if task>=1:
            for epoch in range(args.total_epoch):
                #feat_mean = 0
                train(train_loader,epoch,task,modelm,task_model)
                #test(test_loader,task,modelm,task_model,task_feat_mean)
                test(test_loader,task,modelm,task_model)
                test_known(test_loader,task,modelm,epoch)
                schedulerG.step()

            #acc1, feat_mean = load_model(task,modelm)
            acc1 = load_model(task,modelm)
            task_model.append(copy.deepcopy(modelm))
            #task_feat_mean.append(feat_mean)
        #++++++++++++++++++++++++++++++#

    task_acc.append(acc1)
    print('Task: '+str(task)+'  Test_accuracy: '+ str(acc1))

# This evaluation only check the accuracy of the current task with all the previous task.
# therefore not correct since if trained the model for the 10 task the 1st task will get
# evaluated with 10 task. Next cell provides the model evaluation

print(task_acc)





# def inference_check_task(dtask, mtask, inputs, task_model):
#     joint_entropy = []
#     with torch.no_grad():
#         for m in range(mtask + 1):
#             task_model[m].eval()
#             outputs, _ = task_model[m](inputs)

#             sout = F.softmax(outputs, 1)
#             entropy = -torch.sum(sout * torch.log(sout + 0.0001), 1)
#             joint_entropy.append(entropy)

#     all_entropy = torch.zeros([joint_entropy[0].shape[0], mtask + 1]).cuda()
#     for i in range(mtask + 1):
#         all_entropy[:, i] = joint_entropy[i]
#     ctask = torch.argmin(all_entropy, axis=1) == dtask
#     correct = sum(ctask)

#     return ctask, correct, all_entropy


# def inferecne(test_loader, dtask, mtask, task_model):
#     global best_acc
#     model = task_model[dtask].eval()
#     test_loss = 0
#     correct = 0
#     total = 0
#     cl_loss = 0
#     tcorrect = 0
#     accuracy = []
#     with torch.no_grad():
#         for batch_idx, (inputs, targets) in enumerate(test_loader):
#             inputs, targets1 = inputs.cuda(), targets.cuda()
#             targets = targets1 - dtask * args.class_per_task

#             if mtask > 0:
#                 correct_sample, Ncorrect, _ = inference_check_task(dtask, mtask, inputs, task_model)
#                 tcorrect += Ncorrect
#                 inputs = inputs[correct_sample]
#                 targets = targets[correct_sample]

#             if inputs.shape[0] != 0:
#                 outputs, _ = model(inputs)
#                 loss = criterion(outputs, targets)

#                 test_loss += loss.item()
#                 _, predicted = outputs.max(1)
#                 correct += predicted.eq(targets).sum().item()
#             total += targets1.size(0)
#     if mtask > 0:
#         taskC = tcorrect.item() / total
#     else:
#         taskC = 1.0
#     acc = 100. * correct / total
#     print("[Test Accuracy: %f], [Loss: %f] [Correct: %f]" % (acc, test_loss / (batch_idx + 1), taskC))

#     return acc, taskC





# cl_acc=[]
# criterion = nn.CrossEntropyLoss()
# for mtask in range(args.num_task):
#     accuracy=[]
#     corr=[]
#     print('#######################################################')
#     ttask=mtask+1
#     for dtask in range(ttask):
#         print('Testing Task :---'+str(dtask))
#         test_loader = task_data[dtask][1]
#         acc,correct=inferecne(test_loader,dtask,mtask,task_model)
#         accuracy.append(acc)
#         corr.append(correct)
#     task_acc=np.mean(accuracy)
#     task_correct=np.mean(corr)
#     print('acc: '+str(task_acc)+'  Correct:  '+str(task_correct))
#     cl_acc.append(task_acc)
# print(np.array(cl_acc))



print("\nend time : {}".format(time.time()))