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

##Relevant graphs
- To check our implementation, we have trained 30 different models for Raspberry Pi with varying values of alpha, beta, gamma and delta
- Each colour on the bubble chart represents a particular model and the size of the bubble represents accuracy.
![alpha-beta-latency-accuracy]
(https://github.com/hpnair/18663_Project_FBNet/blob/master/Alpha%2C%20Beta%20vs%20Latency%20.png)
![gamma-delta-energy-accuracy](https://github.com/hpnair/18663_Project_FBNet/blob/master/Gamma%2C%20Delta%2C%20Energy.png)
![latency-energy-accuracy](https://github.com/hpnair/18663_Project_FBNet/blob/master/Latency%2C%20Energy%20vs%20Accuracy%20(Model%2029).png)

##Webpage for Training
![project-website]
(https://github.com/hpnair/18663_Project_FBNet/blob/master/project_hanna_website.png)
##Example Tensorboard visualization
![Tensorboard-log-visualization]
(https://github.com/hpnair/18663_Project_FBNet/blob/master/tensorboard_output_1.png)

