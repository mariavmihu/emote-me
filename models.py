import torch
from torch import nn
from helper_functions import ones_target, zeros_target

# DEFINE THE NEURAL NETWORKS
class DiscriminatorNet(torch.nn.Module):
    """
    A three hidden-layer discriminative neural network
    
    Discrimator outputs probability that the image is real (??)
    """
    def __init__(self, image_size):
        super(DiscriminatorNet, self).__init__()

        self.ngpu = 1
        
        num_channels = 3 #will always be 3 for coloured image (RGB)
        features =  image_size
        
        self.hidden0 = nn.Sequential( 
            nn.Conv2d(num_channels, features, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.2) #prevents overfitting ---> tutorial didnt have this, potential painpoint
        )
        self.hidden1 = nn.Sequential(
            nn.Conv2d(features, features*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features*2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.2) #prevents overfitting ---> tutorial didnt have this, potential painpoint
        )
        self.hidden2 = nn.Sequential(
            nn.Conv2d(features*2, features*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features*4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.2) #prevents overfitting ---> tutorial didnt have this, potential painpoint
        )
        self.hidden3 = nn.Sequential(
            nn.Conv2d(features*4, features*8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features*8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.2) #prevents overfitting ---> tutorial didnt have this, potential painpoint
        )        
        self.out = nn.Sequential(
            nn.Conv2d(features*8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x): #this is basically passing the value through the structure of the network
        x = self.hidden0(x)
        x = self.hidden1(x)
        x = self.hidden2(x)
        x = self.hidden3(x)
        x = self.out(x)
        return x

    def train_discriminator(self, optimizer, real_data, fake_data, loss):
        N = real_data.size(0)
        # Reset gradients
        optimizer.zero_grad()
        
        # 1.1 Train on Real Data
        prediction_real = self.forward(real_data)
        # Calculate error and backpropagate
        error_real = loss(prediction_real, ones_target(N, self.ngpu) )
        error_real.backward()

        # 1.2 Train on Fake Datas
        fake_data.to("cpu")
        prediction_fake = self.forward(fake_data)
        # Calculate error and backpropagate
        error_fake = loss(prediction_fake, zeros_target(N, self.ngpu))
        error_fake.backward()
        
        # 1.3 Update weights with gradients
        optimizer.step()
        
        # Return error and predictions for real and fake inputs
        return error_real + error_fake, prediction_real, prediction_fake

class GeneratorNet(torch.nn.Module):
    """
    A three hidden-layer generative neural network
    """
    def __init__(self, image_size):
        super(GeneratorNet, self).__init__()
        
        self.ngpu = 1
        self.device = torch.device("cuda:0" if (torch.cuda.is_available() and self.ngpu > 0) else "cpu")
        
        num_channels = 3    #always 3 channels because we are working with coloured images (RGB)
        latent_vector_size = 100 #called nz in the tutorial
        features = image_size #called ngf in the tutorial
        
        self.hidden0 = nn.Sequential(
            #ConvTranspose2d(in_channels, out_channels, kernel_size, stride=1, padding=0, 
            #                output_padding=0, groups=1, bias=True, dilation=1, padding_mode='zeros', device=None, dtype=None)
            nn.ConvTranspose2d(latent_vector_size, features*8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(features * 8),
            nn.LeakyReLU(0.2, inplace=True) # ----> tutorial used regular ReLU, potential painpoint??
        )
        
        self.hidden1 = nn.Sequential( 
            nn.ConvTranspose2d(features*8, features*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features * 4),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.hidden2 = nn.Sequential(
            nn.ConvTranspose2d(features*4, features*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features * 2),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        self.hidden3 = nn.Sequential(
            nn.ConvTranspose2d(features*2, features, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        self.out = nn.Sequential(
            nn.ConvTranspose2d(features, num_channels, 4, 2, 1, bias=False),
            nn.Tanh() #maps the resulting values in the [-1,1] range since that is what is done to the input ("real") data as well
        )

    def forward(self, x):
        x = self.hidden0(x)
        x = self.hidden1(x)
        x = self.hidden2(x)
        x = self.hidden3(x)
        x = self.out(x)
        return x

    def train_generator(self,optimizer, fake_data, discriminator, loss):
        N = fake_data.size(0)
        # Reset gradients
        optimizer.zero_grad()
        # Sample noise and generate fake data
        prediction = discriminator(fake_data)
        # Calculate error and backpropagate
        error = loss(prediction, ones_target(N, self.ngpu))
        error.backward()
        # Update weights with gradients
        optimizer.step()
        # Return error
        return error