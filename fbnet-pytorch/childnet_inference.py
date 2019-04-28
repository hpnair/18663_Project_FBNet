import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import argparse
import time
import torchvision
import torchvision.transforms as transforms

import os

parser = argparse.ArgumentParser(description="Train a child net using a theta file.")
parser.add_argument('--theta-folder', type=str, default='theta-folder',
                    help='theta file for selcting blocks for childnet.'
                    )
args = parser.parse_args()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch


print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])


trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=256, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=2)


class ChannelShuffle(nn.Module):
    def __init__(self, group=1):
        assert group > 1
        super(ChannelShuffle, self).__init__()
        self.group = group
    def forward(self, x):
        """https://github.com/Randl/ShuffleNetV2-pytorch/blob/master/model.py
        """
        batchsize, num_channels, height, width = x.data.size()
        assert (num_channels % self.group == 0)
        channels_per_group = num_channels // self.group
        # reshape
        x = x.view(batchsize, self.group, channels_per_group, height, width)
        # transpose
        # - contiguous() required if transpose() is used before view().
        #   See https://github.com/pytorch/pytorch/issues/764
        x = torch.transpose(x, 1, 2).contiguous()
        # flatten
        x = x.view(batchsize, -1, height, width)
        return x

class IdentityBlock(nn.Module):
    def __init__(self):
        super(IdentityBlock, self).__init__()
    def forward(self, x):
        return x


class FBNetBlock(nn.Module):

    def __init__(self, C_in, C_out, kernel_size, stride,
              expansion, group):
        super(FBNetBlock, self).__init__()
        if kernel_size == 3:
            padding = 1
        elif kernel_size == 5:
            padding = 2
        else:
            raise ValueError("Not supported kernel_size %d" % kernel_size)
        bias_flag = True
        if group == 1:
            self.op = nn.Sequential(
            nn.Conv2d(C_in, C_in*expansion, 1, stride=1, padding=0,
                  groups=group, bias=bias_flag),
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in*expansion, C_in*expansion, kernel_size, stride=stride, 
                  padding=padding, groups=C_in*expansion, bias=bias_flag),
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in*expansion, C_out, 1, stride=1, padding=0, 
                  groups=group, bias=bias_flag)
            )
        else:
            self.op = nn.Sequential(
            nn.Conv2d(C_in, C_in*expansion, 1, stride=1, padding=0,
                  groups=group, bias=bias_flag),
            nn.ReLU(inplace=False),
            ChannelShuffle(group),
            nn.Conv2d(C_in*expansion, C_in*expansion, kernel_size, stride=stride, 
                  padding=padding, groups=C_in*expansion, bias=bias_flag),
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in*expansion, C_out, 1, stride=1, padding=0, 
                  groups=group, bias=bias_flag),
            ChannelShuffle(group)
            )
        res_flag = ((C_in == C_out) and (stride == 1))
        self.res_flag = res_flag

    def forward(self, x):
        if self.res_flag:
            return self.op(x) + x
        else:
            return self.op(x) # + self.trans(x)


class ChildNet(nn.Module):
    
    def __init__(self, theta_f='../test_folder/' + '_theta_epoch_91.txt'):
        super(ChildNet, self).__init__()
        with open(theta_f) as f:
            block_nos = []
            for layer,line in enumerate(f):
                l = []
                for value in line.split(' '):
                    l.append(value)
                block_nos.append(l.index(max(l)))
        #print('Layer: {} Block: {}'.format(layer+1,block+1))
        
        self.arch = []
        block_list = []
        
        expansion = [1, 1, 3, 6, 1, 1, 3, 6]
        kernel = [3, 3, 3, 3, 5, 5, 5, 5]
        group = [1, 2, 1, 1, 1, 2, 1, 1]
        
        in_ch = [16, 16, 24, 32, 64, 112, 184]
        out_ch = [16, 24, 32, 64, 112, 184, 352]
        n = [1, 4, 4, 4, 4, 4, 1]
        stride = [1, 2, 2, 1, 1, 1, 1]
        
        block_list.append(nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1))
        
        in_ch = 16
        count = 0
        for i in range(0,len(n)):
            stride_p = stride[i]
            for j in range(0,n[i]):
                k = block_nos[count]
                if k==8:
                    block_list.append(IdentityBlock())
                else:
                    block_list.append(FBNetBlock(C_in=in_ch, C_out=out_ch[i], 
                                                 kernel_size=kernel[k], stride=stride_p, 
                                                 expansion=expansion[k], group=group[k]))
                in_ch = out_ch[i]
                stride_p = 1
                count = count + 1
        
        block_list.append(nn.Conv2d(in_channels=352, out_channels=1984, kernel_size=1, stride=1))
        
        #print(block_list)
        child_layers  = []
        for block in block_list:
            if isinstance(block, nn.Module):
                child_layers.append(block)
            

        self.arch = nn.Sequential(*child_layers)

        self.dropout = nn.Dropout(p=0.4, inplace=False)
        
        self.avgpool = nn.AvgPool2d(kernel_size=8, stride=1)
        
        self.fc = nn.Linear(in_features=1984, out_features=10)
        

        
    def forward(self, x):
         
        x = self.arch(x)
        x = self.dropout(x)
        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        
        return x


print('==> Building model..')

def test(net,theta_f,times):

    global best_acc
    net.eval()
    average_time = 0
    with torch.no_grad():

        for batch_idx, (inputs, targets) in enumerate(testloader):
            if (batch_idx < times):

                inputs, targets = inputs.to(device), targets.to(device)
                print('Starting inference: ')
                time.sleep(2)
                start_time = time.time()
                outputs = net(inputs)
                end_time = time.time()
                print('Write average current: ')
                time.sleep(5)
                average_time += (end_time-start_time)
                print('Inference took: ', (end_time-start_time), 's')
                time.sleep(5)
            else:
            	break
        average_time /= batch_idx

        print('Average Inference time for Model: ', theta_f , 'is: ', average_time ,'s')


for theta_f in sorted(os.listdir(args.theta_folder)):
    theta_f =  args.theta_folder + '/' + theta_f
    if theta_f.endswith('.txt'):
	    print(theta_f)
	    net = ChildNet(theta_f)
	    test(net,theta_f,1)

