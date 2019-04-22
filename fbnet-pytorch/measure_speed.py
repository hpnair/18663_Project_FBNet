import time
import torch

def measure(blocks, 
            input_shape = (1, 3, 108, 108),
            result_path='speed_custom.txt'):

  times = 2000
  f = open(result_path, 'w')

  for b in blocks:
    if isinstance(b, list):
      for net in b:
        net.cuda()
        input = torch.randn(input_shape).cuda()
        output = net(input)

        tic = time.time()
        for _ in range(times):
          output = net(input)
        toc = time.time()
        speed = 1.0 * (toc - tic) / times

        f.write('%.7f ' % speed)
        
      f.write('\n')
    else:
      input = torch.randn(input_shape).cuda()
      b.cuda()
      output = b(input)
    input_shape = output.size()
  f.close()


from blocks_custom import get_blocks

blks = get_blocks(face=True)
measure(blks)
        