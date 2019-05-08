### Train cifar10
```shell
python train_cifar10.py --batch-size 32 --log-frequence 100 --warmup 2 --epochs 90 --alpha 0.2 --beta -0.2 --gamma 0.6 --delta 0.6 --energy-file energy.txt --latency-file latency.txt -tb-logs model_name
```

### Train CIFAR10 over webpage
You could also run the program using the webpage implementation provided in this repo. 
Follow these steps:
```shell
npm install 
node app.js
```
**Note:** I use + not * in loss.
