# Neural Architecture Search using FBnets for Raspberry pi

## FBNet 
- Hardware-Aware Efficient ConvNet Design via Differentiable Neural Architecture Search
**Implementation of [FBNet](https://arxiv.org/pdf/1812.03443.pdf) with PyTorch**

**Note:** I use + not * in loss.

## Train cifar10
```shell
python train_cifar10.py --batch-size 32 --log-frequence 100 --warmup 2 --epochs 90 --alpha 0.2 --beta -0.2 --gamma 0.6 --delta 0.6 --energy-file energy.txt --latency-file latency.txt -tb-logs model_name
```

