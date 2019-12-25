import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'          # Specifying GPU

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import random
import numpy as np
from architecture import DenseResNet

cudnn.enabled   = True
cudnn.benchmark = True

num_epoch = 10000
lr_init = 2e-4
lr_stepsize = 5000
batch_train = 2

num_classes= 4
crop_size = (64, 64, 64)

checkpoint_name= 'model_epoch'

print('@%s:  ' % os.path.basename(__file__))

if 1:
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled   = True
    print ('\tset cuda environment')
    print ('\t\ttorch.__version__              =', torch.__version__)
    print ('\t\ttorch.version.cuda             =', torch.version.cuda)
    print ('\t\ttorch.backends.cudnn.version() =', torch.backends.cudnn.version())
    try:
        print ('\t\tos[\'CUDA_VISIBLE_DEVICES\']     =',os.environ['CUDA_VISIBLE_DEVICES'])
        NUM_CUDA_DEVICES = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
    except Exception:
        print ('\t\tos[\'CUDA_VISIBLE_DEVICES\']     =','None')
        NUM_CUDA_DEVICES = 1

    print ('\t\ttorch.cuda.device_count()      =', torch.cuda.device_count())
    print ('\t\ttorch.cuda.current_device()    =', torch.cuda.current_device())


print('')
