from __future__ import print_function
#%matplotlib inline
from PIL.Image import MAX_IMAGE_PIXELS
import torch
from torch import nn, optim
from torch.autograd.variable import Variable
from torchvision import transforms, datasets
from utils import Logger
from models import DiscriminatorNet, GeneratorNet
import sys


import argparse
import os
import random
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML


# FOR CREATING RANDOM NOISE
def noise(size):
    '''
    Generates a 1-d vector of gaussian sampled random values
    '''
    #torch.randn(size,size, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False)
    # dtype = datatype of returned tensor
    # layout = layout of returned tensor
    n = Variable(torch.randn(size, 100)) #pytorch tutorial also passed 1,1 , will have to see if that will be necessary
    return n

# DEFINE HELPER FUNCTIONS
# flatten image (to flatten a matrix is to collapse into one dimension)
def images_to_vectors(images):
    return images.view(images.size(0), 784)

# expand flattened image back into 2 dimensions
def vectors_to_images(vectors):
    return vectors.view(vectors.size(0), 1, 28, 28)


if __name__ == "__main__":
    
    '''
        STEP ONE: LOAD THE DATA
    '''
    # I really dont like these being local variables, should probably move them elsewhere
    image_size = 64
    workers = 2
    ngpu = 1
    batch_size = 128
    
    #im fine with this being here though
    _DATA_PATH = 'C:\\Users\\mvmih\\OneDrive\\Desktop\\Personal Projects\\cartoon_gan_project\\emote-me\\data'
    
    # LOOK INTO ADDING MORE TRANSFORMS BASED ON COMMON MANIPULATATIONS DONE FOR GANS
    dataset = dset.ImageFolder(root=_DATA_PATH,
                            transform=transforms.Compose([
                                transforms.Resize(image_size),
                                transforms.CenterCrop(image_size),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                            ]))
    # Create the dataloader
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                            shuffle=True, num_workers=workers)
    num_batches = len(data_loader)

    # Decide which device we want to run on
    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

    '''
        STEP TWO: DEFINE THE NETWORKS
    '''
    discriminator = DiscriminatorNet(image_size)
    generator = GeneratorNet(image_size)
    
    '''
        STEP THREE: DEFINE OPTIMIZERS
    '''
    #pytorch tutorial had a beta1 parameter, beta1 = 0.5 and then Adam was initialized with the extra parameter betas=(beta1, 0.999)
    d_optimizer = optim.Adam(discriminator.parameters(), lr=0.0002) 
    g_optimizer = optim.Adam(generator.parameters(), lr=0.0002)
    
    '''
        STEP FOUR: DEFINE LOSS --> this can remain the same, the individual training setups 
                                    within the two models may just need to change
    '''
    loss = nn.BCELoss() 
    
    '''
        STEP FIVE: TRAIN BOTH MODELS
    '''
    num_test_samples = 16
    test_noise = noise(num_test_samples)
    
    # Create logger instance
    logger = Logger(model_name='DCGAN', data_name='Chibi Faces')
    # Total number of epochs to train
    num_epochs = 200 #the one in the pytorch tutorial had only 5
    
    for epoch in range(num_epochs):
        for n_batch, (real_batch,_) in enumerate(data_loader):
            N = real_batch.size(0)
            # 1. Train Discriminator
            real_data = Variable(images_to_vectors(real_batch))
            # Generate fake data and detach 
            # (so gradients are not calculated for generator)
            fake_data = generator(noise(N)).detach()
            # Train D
            d_error, d_pred_real, d_pred_fake = \
                discriminator.train_discriminator(d_optimizer, real_data, fake_data, loss)

            # 2. Train Generator
            # Generate fake data
            fake_data = generator(noise(N))
            # Train G
            g_error = generator.train_generator(g_optimizer, fake_data, discriminator, loss)
            # Log batch error
            logger.log(d_error, g_error, epoch, n_batch, num_batches)
            # Display Progress every few batches
            if (n_batch) % 100 == 0: 
                test_images = vectors_to_images(generator(test_noise))
                test_images = test_images.data
                logger.log_images(
                    test_images, num_test_samples, 
                    epoch, n_batch, num_batches
                );
                # Display status Logs
                logger.display_status(
                    epoch, num_epochs, n_batch, num_batches,
                    d_error, g_error, d_pred_real, d_pred_fake
                )