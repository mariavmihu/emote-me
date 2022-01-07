import torch
from torch.autograd.variable import Variable

# the target output from the discriminator for REAL images are 1s
def ones_target(size):
    '''
    Tensor containing ones, with shape = size
    '''
    data = Variable(torch.ones(size, 1))
    return data

# the target output from the discriminator for FAKE images are 0s
def zeros_target(size):
    '''
    Tensor containing zeros, with shape = size
    '''
    data = Variable(torch.zeros(size, 1))
    return data
