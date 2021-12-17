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

### tSNE
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
    model_path = 'ckpt816/cifar100-10-tSNE-npc-l1-save'     ### change

    #+++ additive angular margin loss +++#
    scale = 32      # exp range[0, 8.8861e+6, 7.8964e+13]
    margin = 0.9396 # cos e, e is 20 degree
    l = 1           # ratio of intra and inter task loss
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




### entropy check
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

### class mean check
# def check_task(task, inputs, model, task_model):
#     global class_mean
#     cosine_list = []
#     with torch.no_grad():
#         for m in range(task):
#             task_model[m].eval()
#             outputs, feat_prev = task_model[m](inputs)
#             nfeat_prev = F.normalize(feat_prev)

#             for i in range(m * args.class_per_task, (m+1) * args.class_per_task):
#                 class_mean_prev = class_mean[i].view(1, -1).cuda()
#                 nclass_mean_prev = F.normalize(class_mean_prev)
#                 cosine_list.append(nfeat_prev @ nclass_mean_prev.transpose(0,1))
            
#         model.eval()
#         outputs, feat_current = model(inputs)
#         nfeat_current = F.normalize(feat_current)

#         for i in range(task * args.class_per_task, (task+1) * args.class_per_task):
#             class_mean_current = class_mean[i].view(1, -1).cuda()
#             nclass_mean_current = F.normalize(class_mean_current)
#             cosine_list.append(nfeat_current @ nclass_mean_current.transpose(0,1))
            
#     cosine = torch.cat(cosine_list, dim=1)
#     ctask = torch.argmax(cosine, dim=1) // args.class_per_task == task
#     correct = sum(ctask)

#     #print(torch.argmax(cosine, dim=1))

#     return ctask, correct, cosine

def test(test_loader, task, model, task_model):
    model.eval()
    feat_list = []
    targets_list = []
    fig = plt.figure(figsize = (10,10))
    plt.axis('off')
    sns.set_style('darkgrid')

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets1 = inputs.cuda(), targets.cuda()
            targets = targets1 - task * args.class_per_task
            if task > 0:
                correct_sample, Ncorrect, _ = check_task(task, inputs, model, task_model)

                #print(Ncorrect)

    #print('\n')

                # correct
                _, feat_c = model(inputs[correct_sample])
                
                feat_list.append(feat_c.cpu())
                targets_list.append(torch.ones(feat_c.shape[0]) * 1)

                # wrong
                _, feat_w = model(inputs[~correct_sample])
                
                feat_list.append(feat_w.cpu())
                targets_list.append(torch.ones(feat_w.shape[0]) * 0)

    feat_all = torch.cat(feat_list, dim=0)
    targets_all = torch.cat(targets_list, dim=0)
    feat_tsne = tsne.fit_transform(feat_all.data)
    sns.scatterplot(feat_tsne[:,0], feat_tsne[:,1], hue=targets_all, legend='full', palette=sns.color_palette("bright", 2))
    plt.legend(['wrong', 'correct'])
    #plt.savefig('./tsne/npcl1.{}.cmcorrect.png'.format(task))    ### change





# groupwise and pointwise group convolutional size
gp=8
pt=16
###############################################

task_model=[]
task_acc=[]

for task in range(10): # range(args.num_task):
    modelm = Net().cuda()
    acc1 = load_model(task,modelm)
    task_model.append(copy.deepcopy(modelm))
    task_acc.append(acc1)



# ### get class mean
# class_feat = [[] for _ in range(args.num_class)]
# class_mean = [0 for _ in range(args.num_class)]

# for m in range(args.num_task):
#     train_loader = task_data[m][0]
#     task_model[m].eval()
#     with torch.no_grad():
#         for batch_idx, (inputs, targets) in enumerate(train_loader):
#             inputs, targets = inputs.cuda(), targets.cuda()

#             outputs, feat = task_model[m](inputs)

#             for i in range(feat.shape[0]):
#                 class_feat[targets[i]].append(feat[i])

# for i in range(args.num_class):
#     class_mean[i] = sum(class_feat[i]) / len(class_feat[i])



## tSNE : wrong samples
task_to_plot = [1, 2, 3, 4, 5, 6, 7, 8, 9]
for m in task_to_plot:
    test_loader = task_data[m][1]
    test(test_loader, m, task_model[m], task_model)



print(task_acc)



print("\nend time : {}".format(time.time()))