# NAS(Neural Architecture Search)

## FBNet 
- Hardware-Aware Efficient ConvNet Design via Differentiable Neural Architecture Search
**Implementation of [FBNet](https://arxiv.org/pdf/1812.03443.pdf) with PyTorch**

**Note:** I use + not * in loss.

## Train cifar10
```shell
python train_cifar10.py --batch-size 32 --log-frequence 100
```

## Train ImageNet
Randomly choose 100 classes from 1000.
You need specify the root dir `imagenet_root` of ImageNet in `train.py`.
```shell
python train.py --batch-size $[24*8] --log-frequence 100 --gpus 0,1,2,3,4,5,6,7
```

**The parameters which are not mentioned in paper:**

- initial value of theta
- cosine decay step, step multiplier, lr multiplier, alpha, ...

**speed**
- `speed_cpu.txt`: measure in cpu
- `speed.txt`: measure in 1080Ti gpu
