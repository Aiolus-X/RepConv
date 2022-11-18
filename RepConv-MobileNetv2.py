import torch
import numpy as np
import math
import time
import random
import os
import torch.nn as nn
from torchvision import datasets, transforms
from torch.autograd import Variable
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from torch.nn import functional as F
from tqdm import tqdm
import logging
from torch.nn.parameter import Parameter
seed = 42

#num_epochs = [0,30,30,30,30]
start_epoch = 1
exp_file = './exp.log'

#data_root = '/data/dataset/ImageNet2012'
data_root = '/data0/imagenet'
batch_size = 384
input_size = 224

def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s]%(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])
 
    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
 
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)
 
    return logger

logger = get_logger(exp_file)


traindir = os.path.join(data_root, 'train')
valdir = os.path.join(data_root, 'val')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_num_threads(8)
np.random.seed(seed)
torch.manual_seed(seed)
dtype = np.float32
kwargs = {"num_workers": 16, "pin_memory": True}
train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225)),
        ]))

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,**kwargs)
val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225)),
        ])),
        batch_size=batch_size, shuffle=False,**kwargs)

class RepConv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=None, groups=1,
                 map_k=3):
        super(RepConv, self).__init__()
        assert map_k <= kernel_size
        self.origin_kernel_shape = (out_channels, in_channels // groups, kernel_size, kernel_size)
        self.register_buffer('weight', torch.zeros(*self.origin_kernel_shape))
        G = in_channels * out_channels // (groups ** 2)
        self.num_2d_kernels = out_channels * in_channels // groups
        self.kernel_size = kernel_size
        self.convmap = nn.Conv2d(in_channels=self.num_2d_kernels,
                                 out_channels=self.num_2d_kernels, kernel_size=map_k, stride=1, padding=map_k // 2,
                                 groups=G, bias=False)
        #nn.init.zeros_(self.convmap.weight)
        self.bias = None#nn.Parameter(torch.zeros(out_channels), requires_grad=True)     # must have a bias for identical initialization
        self.stride = stride
        self.groups = groups
        if padding is None:
            padding = kernel_size // 2
        self.padding = padding

    def forward(self, inputs):
        origin_weight = self.weight.view(1, self.num_2d_kernels, self.kernel_size, self.kernel_size)
        kernel = self.weight + self.convmap(origin_weight).view(*self.origin_kernel_shape)
        return F.conv2d(inputs, kernel, stride=self.stride, padding=self.padding, dilation=1, groups=self.groups, bias=self.bias)


class Block(nn.Module):
    '''expand + depthwise + pointwise'''
    def __init__(self, in_planes, out_planes, expansion, stride):
        super(Block, self).__init__()
        self.stride = stride

        planes = expansion * in_planes
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = RepConv(planes, planes, kernel_size=3, stride=stride, padding=None, groups=planes, map_k=3)
        #nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, groups=planes, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_planes)

        self.shortcut = nn.Sequential()
        if stride == 1 and in_planes != out_planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = out + self.shortcut(x) if self.stride==1 else out
        return out


class MobileNetV2(nn.Module):
    # (expansion, out_planes, num_blocks, stride)
    cfg = [(1,  16, 1, 1),
           (6,  24, 2, 2),  # NOTE: change stride 2 -> 1 for CIFAR10
           (6,  32, 3, 2),
           (6,  64, 4, 2),
           (6,  96, 3, 1),
           (6, 160, 3, 2),
           (6, 320, 1, 1)]

    def __init__(self, num_classes=1000):
        super(MobileNetV2, self).__init__()
        # NOTE: change conv1 stride 2 -> 1 for CIFAR10
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.layers = self._make_layers(in_planes=32)
        self.conv2 = nn.Conv2d(320, 1280, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(1280)
        self.linear = nn.Linear(1280, num_classes)

    def _make_layers(self, in_planes):
        layers = []
        for expansion, out_planes, num_blocks, stride in self.cfg:
            strides = [stride] + [1]*(num_blocks-1)
            for stride in strides:
                layers.append(Block(in_planes, out_planes, expansion, stride))
                in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layers(out)
        out = F.relu(self.bn2(self.conv2(out)))
        out   = F.adaptive_avg_pool2d(out, 1)
        out = out.view(out.size(0), -1)
        #print(out.size())
        out = self.linear(out)
        return out

net = MobileNetV2()
net = nn.DataParallel(net, device_ids=[0,1,2,3])
net.cuda()
net_dict = torch.load('./k3.pth')
net.module.load_state_dict(net_dict,strict=False)


logger.info('start training!')

weight_decay = 4e-5
lr = 1e-1
num_epochs=100

train_loss=[]
val_loss=[]
train_acc = []
val_acc = []

criterion = nn.CrossEntropyLoss().cuda()

optimizer = torch.optim.SGD(net.parameters(), momentum=0.9,lr=lr,weight_decay = weight_decay)
scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=5e-6, end_factor=1.0, total_iters=5*len(train_loader), last_epoch=-1)
#5-epoch warmup, initial value of 0.1 and cosine annealing for 100 epochs. 
for epoch in range(5):
    net.train()
    start_time = time.time()
    c1=[]
    total=0
    correct1=0
    for i, (x, y) in enumerate(tqdm(train_loader)):
        x = Variable(x)
        y = Variable(y)
        x=x.cuda()
        y=y.cuda()
        output = net(x)
        loss1 = criterion(output,y)
        optimizer.zero_grad()
        loss1.backward()
        optimizer.step()
        scheduler.step()
        #lrs.append(optimizer.state_dict()['param_groups'][0]['lr'])
        c1.append(loss1.item())
        total += y.size(0)

        _, predicted = torch.max(output.data, 1)
        correct1 += (predicted == y).sum().item()
                        
    train_loss.append(np.mean(c1))#(torch.mean(torch.stack(c1)))
    t1=100 * correct1 / total
    train_acc.append(t1)
    end_time = time.time()
    #print("Epoch {} loss: {} T1_Accuracy: {}% T5_Accuracy: {}% Time costs: {}s".format(epoch + start_epoch, loss_count[-1], t1, t5, end_time - start_time))
    logger.info("Epoch {} loss: {} T1_Accuracy: {}%  Time costs: {}s".format(epoch + start_epoch, train_loss[-1], t1, end_time - start_time))
    #("Epoch {} Accuracy: {}% Time costs: {}s".format(epoch + start_epoch, t1, end_time - start_time))
    
    net.eval()
    with torch.no_grad():
        c2=[]
        total=0
        correct1=0
        for data in val_loader:
            images, labels = data
            images=images.cuda()
            labels=labels.cuda()
            outputs = net(images).cuda()
            loss2 = criterion(outputs,labels)
            c2.append(loss2.item())
            total += labels.size(0)
                
            _, predicted = torch.max(outputs.data, 1)
            correct1 += (predicted == labels).sum().item()
                
        val_loss.append(np.mean(c2))#(torch.mean(torch.stack(c2)))
        t1=100 * correct1 / total
        val_acc.append(t1)
            
    logger.info('Val_Accuracy:{}%'.format(t1))
    
    torch.save(net.module.state_dict(), './CP.pth')


optimizer = torch.optim.SGD(net.parameters(), momentum=0.9,lr=lr,weight_decay = weight_decay)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=100*len(train_loader), eta_min=5e-7, T_mult=1,last_epoch=-1)

for epoch in range(num_epochs):
    net.train()
    start_time = time.time()
    c1=[]
    total=0
    correct1=0
    for i, (x, y) in enumerate(tqdm(train_loader)):
        x = Variable(x)
        y = Variable(y)
        x=x.cuda()
        y=y.cuda()
        output = net(x)
        loss1 = criterion(output,y)
        optimizer.zero_grad()
        loss1.backward()
        optimizer.step()
        scheduler.step()
        #lrs.append(optimizer.state_dict()['param_groups'][0]['lr'])
        c1.append(loss1.item())
        total += y.size(0)

        _, predicted = torch.max(output.data, 1)
        correct1 += (predicted == y).sum().item()
                        
    train_loss.append(np.mean(c1))#(torch.mean(torch.stack(c1)))
    t1=100 * correct1 / total
    train_acc.append(t1)
    end_time = time.time()
    #print("Epoch {} loss: {} T1_Accuracy: {}% T5_Accuracy: {}% Time costs: {}s".format(epoch + start_epoch, loss_count[-1], t1, t5, end_time - start_time))
    logger.info("Epoch {} loss: {} T1_Accuracy: {}%  Time costs: {}s".format(epoch + start_epoch, train_loss[-1], t1, end_time - start_time))
    #("Epoch {} Accuracy: {}% Time costs: {}s".format(epoch + start_epoch, t1, end_time - start_time))
    
    net.eval()
    with torch.no_grad():
        c2=[]
        total=0
        correct1=0
        for data in val_loader:
            images, labels = data
            images=images.cuda()
            labels=labels.cuda()
            outputs = net(images).cuda()
            loss2 = criterion(outputs,labels)
            c2.append(loss2.item())
            total += labels.size(0)
                
            _, predicted = torch.max(outputs.data, 1)
            correct1 += (predicted == labels).sum().item()
                
        val_loss.append(np.mean(c2))#(torch.mean(torch.stack(c2)))
        t1=100 * correct1 / total
        val_acc.append(t1)
            
    logger.info('Val_Accuracy:{}%'.format(t1))
    
    torch.save(net.module.state_dict(), './CP.pth')

    
logger.info('finish training!')
torch.save(net.module.state_dict(), './tsl.pth')

logger.info(max(val_acc))
logger.info(val_acc[-1])

logger.info(train_acc)
logger.info(val_acc)

logger.info(train_loss)
logger.info(val_loss)
