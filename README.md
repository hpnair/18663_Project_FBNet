# Hardware Aware Neural Network Architecture Search using FBnets for Raspberry pi

## FBNet 
- Hardware-Aware Efficient ConvNet Design via Differentiable Neural Architecture Search
**Implementation of [FBNet](https://arxiv.org/pdf/1812.03443.pdf) with PyTorch**

## Salient features
- Includes energy term along with latency term in the loss function 
- Loss = CE(w,a) + alpha * (Lat(a) ^ beta) + gamma * (ENER(a) ^ delta)
- Use alpha and beta to tune latency related loss.
- Use gamma and delta to tune energy related loss.
- Tested over Raspberry Pi.
- Supports Tensorboard for accuracy, latency and energy visualization as the FbNet trains (logdir = arg provided to "-tb-logs")
- Energy profiling performed using Power Jive.
- Results (on Raspberr Pi 3B):


| Model  | Accuracy(%) | Latency(s) | Energy(J) |
| -----------  | ------| ---------| ---------|
| MobileNetV2  | 75.2  | 7.1 | 18.24 |
| CondenseNet  | 70.5  | 4.83 | 9.28 |
| Ours(HANNA)  | 65.1 | 2.88 | 4.79 |




The following implementation uses stride values that fit for CIFAR10. Feel free to change stride values for ImageNet as mentioned in the paper. We have used 
