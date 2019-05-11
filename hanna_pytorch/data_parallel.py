import torch
from torch.nn.parallel._functions import Broadcast
from torch.nn.parallel.data_parallel import replicate, \
    scatter_kwargs, gather, parallel_apply, _check_balance, \
    _get_device_index


class DataParallel(torch.nn.Module):

  def __init__(self, module, device_ids=None, output_device=None, dim=0):
    super(DataParallel, self).__init__()

    if not torch.cuda.is_available():
        self.module = module
        self.device_ids = []
        return

    if device_ids is None:
        device_ids = list(range(torch.cuda.device_count()))
    if output_device is None:
        output_device = device_ids[0]

    self.dim = dim
    self.module = module
    self.device_ids = list(map(lambda x: _get_device_index(x, True), device_ids))
    self.output_device = _get_device_index(output_device, True)
    self.theta = module.theta

    _check_balance(self.device_ids)

    if len(self.device_ids) == 1:
        self.module.cuda(device_ids[0])

  def forward(self, *inputs, **kwargs):
    batch_size = inputs[0].size()[0]
    self.batch_size = batch_size
    # assert batch_size % len(self.device_ids) == 0

    # Data parallel
    inputs, kwargs = scatter_kwargs(inputs, kwargs, self.device_ids)

    if len(self.device_ids) == 1:
      return self.module(*inputs[0], **kwargs[0])
    else:
      replicas = replicate(self.module, self.device_ids[:len(inputs)])
      theta_list = [[] for _ in self.device_ids]
      for t in self.theta:
        t_ = Broadcast.apply(self.device_ids, t)
        for dev in range(len(self.device_ids)):
          theta_list[dev].append(t_[dev])
      for i, k in enumerate(kwargs):
        k['theta_list'] = theta_list[i]

      outputs = parallel_apply(replicas, 
          inputs, kwargs, self.device_ids[:len(replicas)])
      outputs = gather(outputs, self.device_ids[0])

      return [o.mean() for o in outputs]
