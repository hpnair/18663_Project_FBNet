import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import time
import logging

np.random.seed(0)
sns.set()
from utils import AvgrageMeter, weights_init, \
                  CosineDecayLR, Tensorboard
from data_parallel import DataParallel

class MixedOp(nn.Module):
  """Mixed operation.
  Weighted sum of blocks.
  """
  def __init__(self, blocks):
    super(MixedOp, self).__init__()
    self._ops = nn.ModuleList()
    for op in blocks:
      self._ops.append(op)

  def forward(self, x, weights):
    tmp = []
    for i, op in enumerate(self._ops):
      r = op(x)
      w = weights[..., i].reshape((-1, 1, 1, 1))
      res = w * r
      tmp.append(res)
    return sum(tmp)

class FBNet(nn.Module):

  def __init__(self, num_classes, blocks,
               init_theta=1.0,
               speed_f='./speed.txt',
               energy_f='./energy.txt',
               alpha=0,
               beta=0,
               gamma=0,
               delta=0,
               dim_feature=1984):
    super(FBNet, self).__init__()
    init_func = lambda x: nn.init.constant_(x, init_theta)
    
    self._alpha = alpha
    self._beta = beta
    self._gamma = gamma
    self._delta = delta
    self._criterion = nn.CrossEntropyLoss().cuda()

    self.theta = []
    self._ops = nn.ModuleList()
    self._blocks = blocks

    tmp = []
    input_conv_count = 0
    for b in blocks:
      if isinstance(b, nn.Module):
        tmp.append(b)
        input_conv_count += 1
      else:
        break
    self._input_conv = nn.Sequential(*tmp)
    self._input_conv_count = input_conv_count
    for b in blocks:
      if isinstance(b, list):
        num_block = len(b)
        theta = nn.Parameter(torch.ones((num_block, )).cuda(), requires_grad=True)
        init_func(theta)
        self.theta.append(theta)
        self._ops.append(MixedOp(b))
        input_conv_count += 1
    tmp = []
    for b in blocks[input_conv_count:]:
      if isinstance(b, nn.Module):
        tmp.append(b)
        input_conv_count += 1
      else:
        break
    self._output_conv = nn.Sequential(*tmp)

    # assert len(self.theta) == 22
    with open(speed_f, 'r') as f:

      _speed = f.readlines()
    self._speed = [[float(t) for t in s.strip().split(' ')] for s in _speed]

  ###########################################33
    energy_f = energy_f

    with open(energy_f, 'r') as f:
      _energy = f.readlines()

    self._energy = [[float (t) for t in s.strip().split(' ')] for s in _energy]
#############################################
    # TODO
    max_len = max([len(s) for s in self._speed])
    iden_s = 0.0
    iden_s_c = 0
    for s in self._speed:
      if len(s) == max_len:
        iden_s += s[max_len - 1]
        iden_s_c += 1
    iden_s /= iden_s_c
    for i in range(len(self._speed)):
      if len(self._speed[i]) == (max_len - 1):
        self._speed[i].append(iden_s)
###################################################3
    max_len = max([len(s) for s in self._energy])
    iden_s = 0.0
    iden_s_c = 0
    for s in self._energy:
      if len(s) == max_len:
        iden_s += s[max_len - 1]
        iden_s_c += 1
    iden_s /= iden_s_c
    for i in range(len(self._energy)):
      if len(self._energy[i]) == (max_len - 1):
        self._energy[i].append(iden_s)
#######################################################
    self._speed = torch.tensor(self._speed, requires_grad=False)

#######################################################
    self._energy = torch.tensor(self._energy, requires_grad=False)
#######################################################
    self.classifier = nn.Linear(dim_feature, num_classes)
    # TODO
    # nn.Sequential(nn.BatchNorm2d(dim_feature)
    # nn.Linear(dim_feature, num_classes))

  def forward(self, input, target, temperature=5.0, theta_list=None):
    batch_size = input.size()[0]
    self.batch_size = batch_size
    data = self._input_conv(input)
    theta_idx = 0
    lat = []
    ener = []
    for l_idx in range(self._input_conv_count, len(self._blocks)):
      block = self._blocks[l_idx]
      if isinstance(block, list):
        blk_len = len(block)
        if theta_list is None:
          theta = self.theta[theta_idx]
        else:
          theta = theta_list[theta_idx]
        t = theta.repeat(batch_size, 1)
        weight = nn.functional.gumbel_softmax(t,
                                temperature)
        speed = self._speed[theta_idx][:blk_len].to(weight.device)
        energy = self._energy[theta_idx][:blk_len].to(weight.device)
        lat_ = weight * speed.repeat(batch_size, 1)
        ener_ = weight * energy.repeat(batch_size, 1)
        lat.append(torch.sum(lat_))
        ener.append(torch.sum(ener_))
        data = self._ops[theta_idx](data, weight)
        theta_idx += 1
      else:
        break

    data = self._output_conv(data)
    lat = sum(lat)
    ener = sum(ener)
    data = nn.functional.avg_pool2d(data, data.size()[2:])
    data = data.reshape((batch_size, -1))
    logits = self.classifier(data)

    self.ce = self._criterion(logits, target).sum()
    self.lat_loss = lat / batch_size
    self.ener_loss = ener / batch_size
    self.loss = self.ce +  self._alpha * self.lat_loss.pow(self._beta) + self._gamma * self.ener_loss.pow(self._delta)

    pred = torch.argmax(logits, dim=1)
    # succ = torch.sum(pred == target).cpu().numpy() * 1.0
    self.acc = torch.sum(pred == target).float() / batch_size
    return self.loss, self.ce, self.lat_loss, self.acc, self.ener_loss

class Trainer(object):
  """Training network parameters and theta separately.
  """
  def __init__(self, network,
               w_lr=0.01,
               w_mom=0.9,
               w_wd=1e-4,
               t_lr=0.001,
               t_wd=3e-3,
               t_beta=(0.5, 0.999),
               init_temperature=5.0,
               temperature_decay=0.965,
               logger=logging,
               lr_scheduler={'T_max' : 200},
               gpus=[0],
               save_theta_prefix='',
	       save_tb_log=''):
    assert isinstance(network, FBNet)
    network.apply(weights_init)
    network = network.train().cuda()
    if isinstance(gpus, str):
      gpus = [int(i) for i in gpus.strip().split(',')]
    network = DataParallel(network, gpus)
    self.gpus = gpus
    self._mod = network
    theta_params = network.theta
    mod_params = network.parameters()
    self.theta = theta_params
    self.w = mod_params
    self._tem_decay = temperature_decay
    self.temp = init_temperature
    self.logger = logger
    self.tensorboard = Tensorboard('logs/'+save_tb_log)
    self.save_theta_prefix = save_theta_prefix

    self._acc_avg = AvgrageMeter('acc')
    self._ce_avg = AvgrageMeter('ce')
    self._lat_avg = AvgrageMeter('lat')
    self._loss_avg = AvgrageMeter('loss')
    self._ener_avg = AvgrageMeter('ener')

    self.w_opt = torch.optim.SGD(
                    mod_params,
                    w_lr,
                    momentum=w_mom,
                    weight_decay=w_wd)
    
    self.w_sche = CosineDecayLR(self.w_opt, **lr_scheduler)

    self.t_opt = torch.optim.Adam(
                    theta_params,
                    lr=t_lr, betas=t_beta,
                    weight_decay=t_wd)

  def train_w(self, input, target, decay_temperature=False):
    """Update model parameters.
    """
    self.w_opt.zero_grad()
    loss, ce, lat, acc,ener = self._mod(input, target, self.temp)
    loss.backward()
    self.w_opt.step()
    if decay_temperature:
      tmp = self.temp
      self.temp *= self._tem_decay
      self.logger.info("Change temperature from %.5f to %.5f" % (tmp, self.temp))
    return loss.item(), ce.item(), lat.item(), acc.item(),ener.item()
  
  def train_t(self, input, target, decay_temperature=False):
    """Update theta.
    """
    self.t_opt.zero_grad()
    loss, ce, lat, acc,ener = self._mod(input, target, self.temp)
    loss.backward()
    self.t_opt.step()
    if decay_temperature:
      tmp = self.temp
      self.temp *= self._tem_decay
      self.logger.info("Change temperature from %.5f to %.5f" % (tmp, self.temp))
    return loss.item(), ce.item(), lat.item(), acc.item(),ener.item()
  
  def decay_temperature(self, decay_ratio=None):
    tmp = self.temp
    if decay_ratio is None:
      self.temp *= self._tem_decay
    else:
      self.temp *= decay_ratio
    self.logger.info("Change temperature from %.5f to %.5f" % (tmp, self.temp))
  
  def _step(self, input, target, 
            epoch, step,
            log_frequence,
            func):
    """Perform one step of training.
    """
    input = input.cuda()
    target = target.cuda()
    loss, ce, lat, acc ,ener= func(input, target)

    # Get status
    batch_size = self._mod.batch_size

    self._acc_avg.update(acc)
    self._ce_avg.update(ce)
    self._lat_avg.update(lat)
    self._loss_avg.update(loss)
    self._ener_avg.update(ener)

    if step > 1 and (step % log_frequence == 0):
      self.toc = time.time()
      speed = 1.0 * (batch_size * log_frequence) / (self.toc - self.tic)
      self.tensorboard.log_scalar('Total Loss', self._loss_avg.getValue(), step)
      self.tensorboard.log_scalar('Accuracy',self._acc_avg.getValue(),step)
      self.tensorboard.log_scalar('Latency',self._lat_avg.getValue(),step)
      self.tensorboard.log_scalar('Energy',self._ener_avg.getValue(),step)
      self.logger.info("Epoch[%d] Batch[%d] Speed: %.6f samples/sec %s %s %s %s %s" 
              % (epoch, step, speed, self._loss_avg, 
                 self._acc_avg, self._ce_avg, self._lat_avg,self._ener_avg))
      map(lambda avg: avg.reset(), [self._loss_avg, self._acc_avg, 
                                    self._ce_avg, self._lat_avg,self._ener_avg])
      self.tic = time.time()
  
  def search(self, train_w_ds,
            train_t_ds,
            total_epoch=90,
            start_w_epoch=10,
            log_frequence=100):
    """Search model.
    """
    assert start_w_epoch >= 1, "Start to train w"
    self.tic = time.time()
    for epoch in range(start_w_epoch):
      self.logger.info("Start to train w for epoch %d" % epoch)
      for step, (input, target) in enumerate(train_w_ds):
        self._step(input, target, epoch, 
                   step, log_frequence,
                   lambda x, y: self.train_w(x, y, False))
        self.w_sche.step()
        self.tensorboard.log_scalar('Learning rate curve',self.w_sche.last_epoch,self.w_opt.param_groups[0]['lr'])
        #print(self.w_sche.last_epoch, self.w_opt.param_groups[0]['lr'])

    self.tic = time.time()
    for epoch in range(total_epoch):
      self.logger.info("Start to train theta for epoch %d" % (epoch+start_w_epoch))
      for step, (input, target) in enumerate(train_t_ds):
        self._step(input, target, epoch + start_w_epoch, 
                   step, log_frequence,
                   lambda x, y: self.train_t(x, y, False))
        self.save_theta('./theta-result/%s_theta_epoch_%d.txt' % 
                    (self.save_theta_prefix, epoch+start_w_epoch), epoch)
      self.decay_temperature()
      self.logger.info("Start to train w for epoch %d" % (epoch+start_w_epoch))
      for step, (input, target) in enumerate(train_w_ds):
        self._step(input, target, epoch + start_w_epoch, 
                   step, log_frequence,
                   lambda x, y: self.train_w(x, y, False))
        self.w_sche.step()
      self.tensorboard.close()

  def save_theta(self, save_path='theta.txt',epoch=0):
    """Save theta.
    """
    res = []
    with open(save_path, 'w') as f:
      for i,t in enumerate(self.theta):
        t_list = list(t.detach().cpu().numpy())
        if(len(t_list) < 9): t_list.append(0.00)
        max_index = t_list.index(max(t_list))
        self.tensorboard.log_scalar('Layer %s'% str(i),max_index+1, epoch)
        res.append(t_list)
        s = ' '.join([str(tmp) for tmp in t_list])
        f.write(s + '\n')

      val = np.array(res)
      ax = sns.heatmap(val,cbar=True,annot=True)
      ax.figure.savefig(save_path[:-3]+'png')
      #self.tensorboard.log_image('Theta Values',val,epoch)
      plt.close()
    return res
