import torch
from torch import nn, optim
from torch.autograd.variable import Variable
from torchvision import transforms, datasets
from utils import Logger
from models import DiscriminatorNet, GeneratorNet


# LOADING THE DATA NEEDED --> NOTE: for later, I can create my own dataset using the base classes provided: https://pytorch.org/vision/stable/datasets.html
def mnist_data():
    compose = transforms.Compose(
        [transforms.ToTensor(),
         #parameters for normalization are (mean, standard deviation)
         #the value (0.5) allows you to normalize into the range [-1, 1]
         # this is computed by doing   value = (value - mean) / std   
         #the reason there are 3 values for each is because images have 3 channels, RGB
         #there is a third param to make the operation in-place, but we are not using it
         #transforms.Normalize((.5, .5, .5), (.5, .5, .5)) 
        
        # OKAY BUT IN THIS CASE THE MNIST IMAGES ARE BLACK AND WHITE SO THEY ACTUALLY HAVE ONLY 1 CHANNEL!
         transforms.Normalize([0.5], [0.5]) 
         
        ])
    out_dir = './dataset'
    # what download = True does is check your local directory to see if you already have the dataset downloaded, 
    #  and if you do NOT, it downloads it from the internet
    return datasets.MNIST(root=out_dir, train=True, transform=compose, download=True) 

# FOR CREATING RANDOM NOISE
def noise(size):
    '''
    Generates a 1-d vector of gaussian sampled random values
    '''
    n = Variable(torch.randn(size, 100))
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
    data = mnist_data()
    # Create loader with data, so that we can iterate over it
    data_loader = torch.utils.data.DataLoader(data, batch_size=100, shuffle=True)
    # Num batches
    num_batches = len(data_loader)
    
    '''
        STEP TWO: DEFINE THE NETWORKS
    '''
    discriminator = DiscriminatorNet()
    generator = GeneratorNet()
    
    '''
        STEP THREE: DEFINE OPTIMIZERS
    '''
    d_optimizer = optim.Adam(discriminator.parameters(), lr=0.0002) 
    g_optimizer = optim.Adam(generator.parameters(), lr=0.0002)
    
    '''
        STEP FOUR: DEFINE LOSS
    '''
    loss = nn.BCELoss() 
    
    '''
        STEP FIVE: TRAIN BOTH MODELS
    '''
    num_test_samples = 16
    test_noise = noise(num_test_samples)
    
    # Create logger instance
    logger = Logger(model_name='VGAN', data_name='MNIST')
    # Total number of epochs to train
    num_epochs = 200
    
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