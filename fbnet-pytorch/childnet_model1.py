import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os


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
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)


classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')



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

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
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
    
    def __init__(self):
        super(ChildNet, self).__init__()
        self.layer1 = nn.Sequential(
                          nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1))
        
        self.layer2 = FBNetBlock(C_in=16, C_out=16, kernel_size=3, stride=1, expansion=1, group=2) # Block 2
        
        self.layer3 = FBNetBlock(C_in=16, C_out=24, kernel_size=3, stride=2, expansion=6, group=1) # Block 4
        self.layer4 = FBNetBlock(C_in=24, C_out=24, kernel_size=3, stride=1, expansion=6, group=1) # Block 4
        self.layer5 = FBNetBlock(C_in=24, C_out=24, kernel_size=5, stride=1, expansion=6, group=1) # Block 8
        self.layer6 = FBNetBlock(C_in=24, C_out=24, kernel_size=3, stride=1, expansion=1, group=2) # Block 2
        
        self.layer7 = FBNetBlock(C_in=24, C_out=32, kernel_size=5, stride=2, expansion=6, group=1) # Block 8
        self.layer8 = FBNetBlock(C_in=32, C_out=32, kernel_size=5, stride=1, expansion=1, group=1) # Block 5
        self.layer9 = FBNetBlock(C_in=32, C_out=32, kernel_size=3, stride=1, expansion=1, group=1) # Block 1
        self.layer10 = FBNetBlock(C_in=32, C_out=32, kernel_size=5, stride=1, expansion=1, group=1) # Block 5
        
        self.layer11 = FBNetBlock(C_in=32, C_out=64, kernel_size=3, stride=1, expansion=6, group=1) # Block 4
        self.layer12 = Identity() # Block 9
        self.layer13 = Identity() # Block 9
        self.layer14 = Identity() # Block 9
        
        self.layer15 = FBNetBlock(C_in=64, C_out=112, kernel_size=3, stride=1, expansion=6, group=1) # Block 4
        self.layer16 = Identity() # Block 9
        self.layer17 = Identity() # Block 9
        self.layer18 = Identity() # Block 9
        
        self.layer19 = FBNetBlock(C_in=112, C_out=184, kernel_size=3, stride=1, expansion=1, group=1) # Block 1
        self.layer20 = Identity() # Block 9
        self.layer21 = Identity() # Block 9
        self.layer22 = Identity() # Block 9
        
        self.layer23 = FBNetBlock(C_in=184, C_out=352, kernel_size=3, stride=1, expansion=1, group=1) # Block 1
        
        self.layer24 = nn.Sequential(
                          nn.Conv2d(in_channels=352, out_channels=1984, kernel_size=1, stride=1))
        
        self.avgpool = nn.AvgPool2d(kernel_size=8, stride=1)
        
        self.fc = nn.Linear(in_features=1984, out_features=10)
        
        self.dropout = nn.Dropout(p=0.4, inplace=False)
        
    def forward(self, x):
        x = self.layer1(x)
        
		x = self.layer2(x)

        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)

        x = self.layer7(x)
        x = self.layer8(x)
        x = self.layer9(x)
        x = self.layer10(x)

        x = self.layer11(x)
        x = self.layer12(x)
        x = self.layer13(x)
        x = self.layer14(x)

        x = self.layer15(x)
        x = self.layer16(x)
        x = self.layer17(x)
        x = self.layer18(x)

        x = self.layer19(x)
        x = self.layer20(x)
        x = self.layer21(x)
        x = self.layer22(x)
        
        x = self.layer23(x)

        x = self.layer24(x)

        x = self.dropout(x)
        
		x = self.avgpool(x)
        
		x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        
        return x


print('==> Building model..')
net = ChildNet()
net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True


criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=4e-5)


def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)

        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        print('Batch: {}, Loss: {}, Acc: {}'.format(batch_idx, (train_loss/(batch_idx+1)), 100 * correct/total))


def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            print(total)
            correct += predicted.eq(targets).sum().item()
            print(correct)
            print('Batch: {}, Loss: {}, Acc: {}'.format(batch_idx, (train_loss/(batch_idx+1)), 100 * correct/total))



for epoch in range(start_epoch, start_epoch+90):
    train(epoch)
    test(epoch)

