import torch
from torch import nn
from helper_functions import ones_target, zeros_target

# DEFINE THE NEURAL NETWORKS
class DiscriminatorNet(torch.nn.Module):
    """
    A three hidden-layer discriminative neural network
    
    Discrimator outputs probability that the image is real (??)
    """
    def __init__(self):
        super(DiscriminatorNet, self).__init__()
        n_features = 784
        n_out = 1
        
        self.hidden0 = nn.Sequential( 
            nn.Linear(n_features, 1024),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3) #prevents overfitting
        )
        self.hidden1 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3)
        )
        self.hidden2 = nn.Sequential(
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3)
        )
        self.out = nn.Sequential(
            torch.nn.Linear(256, n_out),
            torch.nn.Sigmoid()
        )

    def forward(self, x): #this is basically passing the value through the structure of the network
        x = self.hidden0(x) # x changes value with pass through the layers until it comes out the end
        x = self.hidden1(x)
        x = self.hidden2(x)
        x = self.out(x)
        return x

    def train_discriminator(self, optimizer, real_data, fake_data, loss):
        N = real_data.size(0)
        # Reset gradients
        optimizer.zero_grad()
        
        # 1.1 Train on Real Data
        prediction_real = self.forward(real_data)
        # Calculate error and backpropagate
        error_real = loss(prediction_real, ones_target(N) )
        error_real.backward()

        # 1.2 Train on Fake Datas
        prediction_fake = self.forward(fake_data)
        # Calculate error and backpropagate
        error_fake = loss(prediction_fake, zeros_target(N))
        error_fake.backward()
        
        # 1.3 Update weights with gradients
        optimizer.step()
        
        # Return error and predictions for real and fake inputs
        return error_real + error_fake, prediction_real, prediction_fake

class GeneratorNet(torch.nn.Module):
    """
    A three hidden-layer generative neural network
    """
    def __init__(self):
        super(GeneratorNet, self).__init__()
        n_features = 100
        n_out = 784
        
        self.hidden0 = nn.Sequential(
            nn.Linear(n_features, 256),
            nn.LeakyReLU(0.2)
        )
        self.hidden1 = nn.Sequential(            
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2)
        )
        self.hidden2 = nn.Sequential(
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2)
        )
        
        self.out = nn.Sequential(
            nn.Linear(1024, n_out), #the output w
            nn.Tanh() #maps the resulting values in the [-1,1] range since that is what is done to the input ("real") data as well
        )

    def forward(self, x):
        x = self.hidden0(x)
        x = self.hidden1(x)
        x = self.hidden2(x)
        x = self.out(x)
        return x

    def train_generator(self,optimizer, fake_data, discriminator, loss):
        N = fake_data.size(0)
        # Reset gradients
        optimizer.zero_grad()
        # Sample noise and generate fake data
        prediction = discriminator(fake_data)
        # Calculate error and backpropagate
        error = loss(prediction, ones_target(N))
        error.backward()
        # Update weights with gradients
        optimizer.step()
        # Return error
        return error